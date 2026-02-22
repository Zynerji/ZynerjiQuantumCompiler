"""Build weighted adjacency matrices from hardware coupling maps.

Converts a Qiskit CouplingMap (and optionally a Target with error rates)
into a weighted adjacency matrix suitable for spectral decomposition.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from qiskit.transpiler import CouplingMap, Target


def build_hardware_graph(
    coupling_map: CouplingMap,
    target: Optional[Target] = None,
    error_weighted: bool = False,
) -> csr_matrix:
    """Build a weighted adjacency matrix from a hardware coupling map.

    Parameters
    ----------
    coupling_map : CouplingMap
        The hardware qubit connectivity.
    target : Target, optional
        Backend target with gate error rate information.
    error_weighted : bool
        If True and target is provided, weight edges by gate fidelity
        (1 - error_rate). Higher fidelity edges get stronger weights,
        encouraging the matcher to place interacting qubits on them.
        If False, all edges have uniform weight 1.0.

    Returns
    -------
    csr_matrix
        Symmetric weighted adjacency matrix (n_qubits x n_qubits).
    """
    n_qubits = coupling_map.size()
    if n_qubits == 0:
        return csr_matrix((0, 0))

    adj = lil_matrix((n_qubits, n_qubits), dtype=np.float64)

    edges = coupling_map.get_edges()

    for q0, q1 in edges:
        if q0 >= n_qubits or q1 >= n_qubits:
            continue

        weight = 1.0
        if error_weighted and target is not None:
            weight = _get_edge_fidelity(target, q0, q1)

        adj[q0, q1] = max(adj[q0, q1], weight)
        adj[q1, q0] = max(adj[q1, q0], weight)

    return adj.tocsr()


def _get_edge_fidelity(target: Target, q0: int, q1: int) -> float:
    """Get gate fidelity (1 - error_rate) for a qubit pair from the Target.

    Falls back to 1.0 if error data is unavailable.
    """
    for gate_name in ("cx", "cz", "ecr", "rzz"):
        if gate_name in target.operation_names:
            try:
                props = target[gate_name].get((q0, q1))
                if props is None:
                    props = target[gate_name].get((q1, q0))
                if props is not None and props.error is not None:
                    return max(0.01, 1.0 - props.error)
            except (KeyError, AttributeError):
                continue
    return 1.0
