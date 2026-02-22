"""Extract weighted interaction graphs from quantum circuits.

Converts a Qiskit DAGCircuit into a weighted adjacency matrix where
edge weights represent the number of two-qubit gates between each pair
of qubits, optionally weighted by gate depth (earlier gates matter more).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit


def build_interaction_graph(
    circuit: QuantumCircuit | DAGCircuit,
    depth_weighted: bool = False,
    exp_decay_tau: float | None = None,
) -> csr_matrix:
    """Build a weighted adjacency matrix from a quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit or DAGCircuit
        The quantum circuit to analyze.
    depth_weighted : bool
        If True, weight edges by 1/(1 + layer_index) so earlier gates
        contribute more. If False, all 2Q gates have weight 1.
    exp_decay_tau : float, optional
        If set, use exponential decay weighting: exp(-layer_idx / tau).
        Gives much sharper focus on early layers than 1/(1+layer).
        Overrides depth_weighted when both are set.

    Returns
    -------
    csr_matrix
        Symmetric weighted adjacency matrix (n_qubits x n_qubits).
    """
    if isinstance(circuit, DAGCircuit):
        dag = circuit
        n_qubits = dag.num_qubits()
    else:
        from qiskit.converters import circuit_to_dag
        dag = circuit_to_dag(circuit)
        n_qubits = circuit.num_qubits

    if n_qubits == 0:
        return csr_matrix((0, 0))

    adj = lil_matrix((n_qubits, n_qubits), dtype=np.float64)

    # Get layers for depth weighting
    if exp_decay_tau is not None or depth_weighted:
        layers = list(dag.layers())
        for layer_idx, layer in enumerate(layers):
            if exp_decay_tau is not None:
                weight = np.exp(-layer_idx / exp_decay_tau)
            else:
                weight = 1.0 / (1.0 + layer_idx)
            for node in layer["graph"].op_nodes():
                qargs = node.qargs
                if len(qargs) == 2:
                    q0 = dag.find_bit(qargs[0]).index
                    q1 = dag.find_bit(qargs[1]).index
                    adj[q0, q1] += weight
                    adj[q1, q0] += weight
    else:
        for node in dag.op_nodes():
            qargs = node.qargs
            if len(qargs) == 2:
                q0 = dag.find_bit(qargs[0]).index
                q1 = dag.find_bit(qargs[1]).index
                adj[q0, q1] += 1.0
                adj[q1, q0] += 1.0

    return adj.tocsr()


def build_signed_interaction_graph(
    circuit: QuantumCircuit | DAGCircuit,
    repulsion_strength: float = 0.3,
) -> csr_matrix:
    """Build a signed adjacency matrix with attraction AND repulsion.

    Inspired by VHGT's ternary {-1, 0, +1} encoding:
    - Positive edges (+weight): qubits that interact directly (2Q gates)
    - Negative edges (-weight): qubits that COMPETE for the same neighbors
      (share interaction partners but don't interact directly)

    The repulsion helps the Laplacian eigenvectors separate competing qubits,
    pushing them to different parts of the hardware. Standard (unsigned)
    interaction graphs can't express "these qubits should be far apart."

    Parameters
    ----------
    circuit : QuantumCircuit or DAGCircuit
        The quantum circuit to analyze.
    repulsion_strength : float
        Strength of repulsive edges relative to attractive ones.
        0.3 means if qubits A and B share a neighbor C with weights
        w(A,C)=2 and w(B,C)=3, the repulsive edge is -0.3 * min(2,3) = -0.6

    Returns
    -------
    csr_matrix
        Signed weighted adjacency matrix (n_qubits x n_qubits).
    """
    # First build the standard (positive) interaction graph
    pos_adj = build_interaction_graph(circuit)
    n = pos_adj.shape[0]

    if n < 3:
        return pos_adj

    # Convert to dense for neighbor analysis
    pos_dense = pos_adj.toarray()

    # Build repulsive edges: for each pair (i, j) with no direct interaction,
    # check if they share interaction partners (compete for neighbors)
    signed = pos_dense.copy()

    for i in range(n):
        neighbors_i = set(np.nonzero(pos_dense[i])[0])
        if not neighbors_i:
            continue
        for j in range(i + 1, n):
            if pos_dense[i, j] > 0:
                continue  # Already have a positive edge, skip
            neighbors_j = set(np.nonzero(pos_dense[j])[0])
            shared = neighbors_i & neighbors_j
            if shared:
                # Compute repulsion strength from shared neighbor weights
                repulsion = 0.0
                for s in shared:
                    repulsion += min(pos_dense[i, s], pos_dense[j, s])
                repulsive_weight = -repulsion_strength * repulsion
                signed[i, j] = repulsive_weight
                signed[j, i] = repulsive_weight

    return csr_matrix(signed)
