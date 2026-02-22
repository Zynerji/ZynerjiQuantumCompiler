"""Graph matching via spectral coordinates and the Hungarian algorithm.

Provides multiple matching strategies:
1. Spectral matching (dual-helix or standard Laplacian)
2. Hall spectral placement (Fiedler x, 3rd eigvec y — from VLSI placement theory)
3. Distance-aware matching (minimize routing cost using hardware shortest paths)
4. Iterative swap refinement (pairwise swap local search)
5. Degree-sorted matching (place high-degree circuit qubits on high-degree hardware qubits)
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh

from qiskit.transpiler import CouplingMap, Layout
from qiskit.circuit import QuantumRegister

from zynerji_qc.core.dual_helix import (
    HelixParams,
    SpectralCoords,
    compute_spectral_coords,
    build_standard_laplacian,
)


def _align_coords(
    circuit_coords: SpectralCoords,
    hardware_coords: SpectralCoords,
) -> tuple[np.ndarray, np.ndarray]:
    """Align spectral coordinate dimensions between circuit and hardware."""
    n_logical = circuit_coords.coords.shape[0]
    n_physical = hardware_coords.coords.shape[0]

    c_dim = circuit_coords.coords.shape[1] if n_logical > 0 else 0
    h_dim = hardware_coords.coords.shape[1] if n_physical > 0 else 0
    max_dim = max(c_dim, h_dim, 1)

    c_coords = np.zeros((n_logical, max_dim))
    if n_logical > 0 and c_dim > 0:
        c_coords[:, :c_dim] = circuit_coords.coords

    h_coords = np.zeros((n_physical, max_dim))
    if n_physical > 0 and h_dim > 0:
        h_coords[:, :h_dim] = hardware_coords.coords

    return c_coords, h_coords


def build_cost_matrix(
    circuit_coords: SpectralCoords,
    hardware_coords: SpectralCoords,
) -> np.ndarray:
    """Build a cost matrix from spectral coordinate distances (L2).

    Returns
    -------
    np.ndarray
        Cost matrix (n_logical, n_physical) where C[i,j] is the squared
        Euclidean distance between circuit qubit i and hardware qubit j
        in spectral space.
    """
    c_coords, h_coords = _align_coords(circuit_coords, hardware_coords)
    n_logical, n_physical = c_coords.shape[0], h_coords.shape[0]

    if n_logical == 0 or n_physical == 0:
        return np.zeros((n_logical, n_physical))

    # Squared Euclidean distance: ||c_i - h_j||^2
    c_sq = np.sum(c_coords ** 2, axis=1, keepdims=True)
    h_sq = np.sum(h_coords ** 2, axis=1, keepdims=True)
    cross = c_coords @ h_coords.T
    cost = c_sq + h_sq.T - 2.0 * cross
    np.maximum(cost, 0.0, out=cost)

    return cost


def build_angular_cost_matrix(
    circuit_coords: SpectralCoords,
    hardware_coords: SpectralCoords,
) -> np.ndarray:
    """Build a cost matrix from angular (cosine) similarity.

    Inspired by VHGT phase alignment: constructive interference (phi~0)
    = good match, destructive (phi~pi) = bad match. Angular cost is
    invariant to magnitude, focusing purely on structural role similarity.

    C[i,j] = 1 - cos(angle between circuit_coords[i] and hardware_coords[j])
    Range: [0, 2] where 0 = identical direction, 2 = opposite.
    """
    c_coords, h_coords = _align_coords(circuit_coords, hardware_coords)
    n_logical, n_physical = c_coords.shape[0], h_coords.shape[0]

    if n_logical == 0 or n_physical == 0:
        return np.zeros((n_logical, n_physical))

    # Normalize to unit vectors
    c_norms = np.linalg.norm(c_coords, axis=1, keepdims=True)
    h_norms = np.linalg.norm(h_coords, axis=1, keepdims=True)
    c_norms = np.maximum(c_norms, 1e-12)
    h_norms = np.maximum(h_norms, 1e-12)

    c_unit = c_coords / c_norms
    h_unit = h_coords / h_norms

    # Cosine similarity matrix
    cos_sim = c_unit @ h_unit.T  # (n_logical, n_physical), range [-1, 1]
    np.clip(cos_sim, -1.0, 1.0, out=cos_sim)

    # Angular cost: 1 - cos(angle) -> 0 for aligned, 2 for anti-aligned
    cost = 1.0 - cos_sim

    return cost


def build_hybrid_cost_matrix(
    circuit_coords: SpectralCoords,
    hardware_coords: SpectralCoords,
    angular_weight: float = 0.5,
) -> np.ndarray:
    """Build a hybrid cost matrix combining L2 distance and angular similarity.

    Blends magnitude-sensitive L2 matching with magnitude-invariant phase matching,
    like VHGT's amplitude * phase decomposition.
    """
    l2_cost = build_cost_matrix(circuit_coords, hardware_coords)
    angular_cost = build_angular_cost_matrix(circuit_coords, hardware_coords)

    if l2_cost.size == 0:
        return l2_cost

    # Normalize both to [0, 1] range before blending
    l2_max = l2_cost.max()
    ang_max = angular_cost.max()
    if l2_max > 1e-12:
        l2_norm = l2_cost / l2_max
    else:
        l2_norm = l2_cost
    if ang_max > 1e-12:
        ang_norm = angular_cost / ang_max
    else:
        ang_norm = angular_cost

    return (1.0 - angular_weight) * l2_norm + angular_weight * ang_norm


def spectral_match(
    circuit_adj: csr_matrix,
    hardware_adj: csr_matrix,
    qreg: QuantumRegister,
    coupling_map: CouplingMap,
    params: Optional[HelixParams] = None,
) -> Layout:
    """Match logical qubits to physical qubits via spectral graph matching."""
    if params is None:
        params = HelixParams()

    n_logical = circuit_adj.shape[0]
    if n_logical == 0:
        return Layout()

    circuit_coords = compute_spectral_coords(circuit_adj, params)
    hardware_coords = compute_spectral_coords(hardware_adj, params)

    cost = build_cost_matrix(circuit_coords, hardware_coords)
    row_ind, col_ind = linear_sum_assignment(cost)

    layout_dict = {}
    for log_idx, phys_idx in zip(row_ind, col_ind):
        if log_idx < n_logical:
            layout_dict[qreg[log_idx]] = phys_idx

    return Layout(layout_dict)


def build_qfactor_cost_matrix(
    circuit_coords: SpectralCoords,
    hardware_coords: SpectralCoords,
    q_amplification: float = 1.5,
    alignment_threshold: float = 0.9,
) -> np.ndarray:
    """Build a cost matrix with Q-factor amplification at structural antinodes.

    Inspired by VHGT's icosahedron/dodecahedron antinode amplification:
    when a circuit qubit's spectral direction strongly aligns with a hardware
    position's spectral direction (antinode), the match is amplified — making
    the Hungarian algorithm more decisive about placing qubits at their
    structurally ideal positions.

    For non-aligned pairs (nodes), the cost is standard L2.
    For aligned pairs (antinodes), the cost is reduced by q_amplification factor.
    """
    c_coords, h_coords = _align_coords(circuit_coords, hardware_coords)
    n_logical, n_physical = c_coords.shape[0], h_coords.shape[0]

    if n_logical == 0 or n_physical == 0:
        return np.zeros((n_logical, n_physical))

    # Standard L2 cost
    c_sq = np.sum(c_coords ** 2, axis=1, keepdims=True)
    h_sq = np.sum(h_coords ** 2, axis=1, keepdims=True)
    cross = c_coords @ h_coords.T
    cost = c_sq + h_sq.T - 2.0 * cross
    np.maximum(cost, 0.0, out=cost)

    # Compute angular alignment (cosine similarity)
    c_norms = np.linalg.norm(c_coords, axis=1, keepdims=True)
    h_norms = np.linalg.norm(h_coords, axis=1, keepdims=True)
    c_norms = np.maximum(c_norms, 1e-12)
    h_norms = np.maximum(h_norms, 1e-12)

    cos_sim = (c_coords / c_norms) @ (h_coords / h_norms).T
    np.clip(cos_sim, -1.0, 1.0, out=cos_sim)

    # Q-factor amplification: reduce cost at antinodes (high alignment)
    # Antinode = |cos_sim| > threshold (both aligned and anti-aligned)
    antinode_mask = np.abs(cos_sim) > alignment_threshold
    cost[antinode_mask] /= q_amplification

    return cost


def angular_spectral_match(
    circuit_adj: csr_matrix,
    hardware_adj: csr_matrix,
    qreg: QuantumRegister,
    coupling_map: CouplingMap,
    params: Optional[HelixParams] = None,
    angular_weight: float = 1.0,
    golden_selection: bool = False,
) -> Layout:
    """Match via angular (phase) similarity in spectral space.

    VHGT-inspired: uses cosine similarity instead of L2 distance.
    Phase alignment is invariant to spectral coordinate magnitude,
    focusing purely on structural role similarity.

    angular_weight: 1.0 = pure angular, 0.5 = hybrid, 0.0 = pure L2.
    """
    if params is None:
        params = HelixParams()

    n_logical = circuit_adj.shape[0]
    if n_logical == 0:
        return Layout()

    circuit_coords = compute_spectral_coords(circuit_adj, params, golden_selection=golden_selection)
    hardware_coords = compute_spectral_coords(hardware_adj, params, golden_selection=golden_selection)

    if angular_weight >= 0.99:
        cost = build_angular_cost_matrix(circuit_coords, hardware_coords)
    elif angular_weight <= 0.01:
        cost = build_cost_matrix(circuit_coords, hardware_coords)
    else:
        cost = build_hybrid_cost_matrix(circuit_coords, hardware_coords, angular_weight)

    row_ind, col_ind = linear_sum_assignment(cost)

    layout_dict = {}
    for log_idx, phys_idx in zip(row_ind, col_ind):
        if log_idx < n_logical:
            layout_dict[qreg[log_idx]] = phys_idx

    return Layout(layout_dict)


def qfactor_spectral_match(
    circuit_adj: csr_matrix,
    hardware_adj: csr_matrix,
    qreg: QuantumRegister,
    coupling_map: CouplingMap,
    params: Optional[HelixParams] = None,
    q_amplification: float = 1.5,
    alignment_threshold: float = 0.85,
    golden_selection: bool = False,
) -> Layout:
    """Match with Q-factor amplification at structural antinodes.

    VHGT-inspired: when circuit and hardware spectral directions align
    strongly (antinode), amplify the match to make the Hungarian algorithm
    more decisive about placing qubits at structurally ideal positions.
    """
    if params is None:
        params = HelixParams()

    n_logical = circuit_adj.shape[0]
    if n_logical == 0:
        return Layout()

    circuit_coords = compute_spectral_coords(circuit_adj, params, golden_selection=golden_selection)
    hardware_coords = compute_spectral_coords(hardware_adj, params, golden_selection=golden_selection)

    cost = build_qfactor_cost_matrix(
        circuit_coords, hardware_coords,
        q_amplification=q_amplification,
        alignment_threshold=alignment_threshold,
    )
    row_ind, col_ind = linear_sum_assignment(cost)

    layout_dict = {}
    for log_idx, phys_idx in zip(row_ind, col_ind):
        if log_idx < n_logical:
            layout_dict[qreg[log_idx]] = phys_idx

    return Layout(layout_dict)


def hall_spectral_match(
    circuit_adj: csr_matrix,
    coupling_map: CouplingMap,
    qreg: QuantumRegister,
    normalized: bool = True,
) -> Layout:
    """Hall 1970 spectral placement adapted for qubit mapping.

    Uses the Fiedler vector (2nd eigenvector) as x-coordinate and the
    3rd eigenvector as y-coordinate for both circuit and hardware graphs.
    Then matches qubits by proximity in this shared 2D spectral space
    using the Hungarian algorithm.

    The normalized Laplacian (L_sym = I - D^{-1/2}AD^{-1/2}) is used
    by default to handle degree-irregular topologies like heavy-hex.
    """
    n_logical = circuit_adj.shape[0]
    n_physical = coupling_map.size()
    if n_logical == 0:
        return Layout()

    hw_adj = _coupling_map_to_adj(coupling_map)

    # Build Laplacians
    L_circ = build_standard_laplacian(circuit_adj, normalized=normalized)
    L_hw = build_standard_laplacian(hw_adj, normalized=normalized)

    # Extract 2nd and 3rd eigenvectors (skip trivial 0th)
    circ_coords = _hall_eigenvectors(L_circ, n_logical)
    hw_coords = _hall_eigenvectors(L_hw, n_physical)

    # Build cost matrix: squared Euclidean distance in 2D spectral space
    # circ_coords: (n_logical, 2), hw_coords: (n_physical, 2)
    c_sq = np.sum(circ_coords ** 2, axis=1, keepdims=True)
    h_sq = np.sum(hw_coords ** 2, axis=1, keepdims=True)
    cross = circ_coords @ hw_coords.T
    cost = c_sq + h_sq.T - 2.0 * cross
    np.maximum(cost, 0.0, out=cost)

    row_ind, col_ind = linear_sum_assignment(cost)

    layout_dict = {}
    for log_idx, phys_idx in zip(row_ind, col_ind):
        if log_idx < n_logical:
            layout_dict[qreg[log_idx]] = phys_idx

    return Layout(layout_dict)


def _hall_eigenvectors(L: csr_matrix, n: int) -> np.ndarray:
    """Extract 2nd and 3rd eigenvectors for Hall placement (n x 2)."""
    if n < 3:
        return np.zeros((n, 2))

    num_request = min(4, n)
    try:
        if n < 10:
            evals, evecs = np.linalg.eigh(L.toarray())
        else:
            evals, evecs = eigsh(L, k=num_request, which="SM", tol=1e-8, maxiter=2000)
        idx = np.argsort(evals)
        # Fiedler = idx[1], 3rd eigvec = idx[2]
        coords = np.zeros((n, 2))
        if len(idx) > 1:
            coords[:, 0] = evecs[:, idx[1]]
        if len(idx) > 2:
            coords[:, 1] = evecs[:, idx[2]]
        return coords
    except Exception:
        return np.zeros((n, 2))


def refine_layout_swaps(
    layout: Layout,
    circuit_adj: csr_matrix,
    coupling_map: CouplingMap,
    qreg: QuantumRegister,
    max_rounds: int = 3,
) -> Layout:
    """Iterative pairwise swap refinement of a layout.

    For each pair of placed qubits, try swapping their physical assignments.
    Keep the swap if it reduces total weighted hardware distance.
    Repeat until no improvement or max_rounds exhausted.

    Adapted from ResonantQ VLSI floorplanning 3-stage refinement.
    """
    n_logical = circuit_adj.shape[0]
    if n_logical < 2:
        return layout

    hw_adj = _coupling_map_to_adj(coupling_map)
    dist_matrix = shortest_path(hw_adj, directed=False, unweighted=True)

    # Extract interaction edges
    circ_coo = circuit_adj.tocoo()
    interactions = []
    for u, v, w in zip(circ_coo.row, circ_coo.col, circ_coo.data):
        if u < v:
            interactions.append((u, v, w))
    if not interactions:
        return layout

    # Current placement: logical -> physical
    placement = {}
    for i in range(n_logical):
        try:
            placement[i] = layout[qreg[i]]
        except (KeyError, IndexError):
            placement[i] = i

    def compute_cost(p):
        """Total weighted routing distance for a placement."""
        total = 0.0
        for u, v, w in interactions:
            if u in p and v in p:
                total += w * dist_matrix[p[u], p[v]]
        return total

    best_cost = compute_cost(placement)

    for _ in range(max_rounds):
        improved = False
        for i in range(n_logical):
            for j in range(i + 1, n_logical):
                # Try swapping physical assignments of logical qubits i and j
                placement[i], placement[j] = placement[j], placement[i]
                new_cost = compute_cost(placement)
                if new_cost < best_cost - 1e-10:
                    best_cost = new_cost
                    improved = True
                else:
                    # Revert
                    placement[i], placement[j] = placement[j], placement[i]
        if not improved:
            break

    layout_dict = {qreg[i]: placement[i] for i in range(n_logical)}
    return Layout(layout_dict)


def distance_aware_match(
    circuit_adj: csr_matrix,
    coupling_map: CouplingMap,
    qreg: QuantumRegister,
) -> Layout:
    """Match by minimizing total routing cost using hardware shortest-path distances.

    For each candidate assignment of logical→physical qubits, the cost is
    the sum over all 2Q gates of the shortest-path distance between the
    assigned physical qubits, weighted by gate count.

    This directly optimizes what routing cares about: how far apart
    interacting qubits are on the hardware.
    """
    n_logical = circuit_adj.shape[0]
    n_physical = coupling_map.size()
    if n_logical == 0:
        return Layout()

    # Hardware shortest-path distance matrix
    hw_adj = _coupling_map_to_adj(coupling_map)
    dist_matrix = shortest_path(hw_adj, directed=False, unweighted=True)

    # Circuit interaction edges (upper triangle)
    circ_coo = circuit_adj.tocoo()
    interactions = []  # (logical_i, logical_j, weight)
    for u, v, w in zip(circ_coo.row, circ_coo.col, circ_coo.data):
        if u < v:
            interactions.append((u, v, w))

    if not interactions:
        # No 2Q gates — degree-sorted fallback
        return _degree_sorted_match(circuit_adj, coupling_map, qreg)

    # Build cost matrix: C[i,j] = cost of placing logical qubit i on physical qubit j
    # For LAP, we compute: for each (log_i, log_j, w), add w * dist(phys_j1, phys_j2)
    # This is a QAP (NP-hard), so we approximate via LAP relaxation:
    # C[i,j] = sum over neighbors k of i: interaction_weight(i,k) * avg_dist(j, neighbors_of_j_in_hw)
    cost = np.zeros((n_logical, n_physical))

    # For each logical qubit, compute weighted average distance to all physical qubits
    for log_i, log_j, w in interactions:
        # If we place log_i at phys_a and log_j at phys_b, cost += w * dist(a,b)
        # LAP relaxation: for each phys candidate for log_i, accumulate expected cost
        for phys in range(n_physical):
            cost[log_i, phys] += w * dist_matrix[phys, :].mean()
            cost[log_j, phys] += w * dist_matrix[phys, :].mean()

    # Better approximation: use interaction-weighted center-of-mass
    # For each logical qubit i, its ideal position minimizes
    # sum_j w(i,j) * dist(pos_i, pos_j)
    # Approximate: place qubit i near the hardware center weighted by its interactions
    cost = np.zeros((n_logical, n_physical))

    # Compute interaction degree for each logical qubit
    interaction_degree = np.zeros(n_logical)
    for log_i, log_j, w in interactions:
        interaction_degree[log_i] += w
        interaction_degree[log_j] += w

    # Hardware centrality (inverse average distance to all other qubits)
    avg_dist = dist_matrix.mean(axis=1)  # (n_physical,)
    # High-interaction qubits should go to low-avg-distance (central) hardware qubits
    for i in range(n_logical):
        cost[i, :] = interaction_degree[i] * avg_dist

    # Add pairwise interaction cost
    for log_i, log_j, w in interactions:
        for p in range(n_physical):
            # Penalize placing log_i on physical qubits far from central ones
            cost[log_i, p] += w * dist_matrix[p, :].min()  # Distance to nearest neighbor

    row_ind, col_ind = linear_sum_assignment(cost)

    layout_dict = {}
    for log_idx, phys_idx in zip(row_ind, col_ind):
        if log_idx < n_logical:
            layout_dict[qreg[log_idx]] = phys_idx
    return Layout(layout_dict)


def greedy_distance_match(
    circuit_adj: csr_matrix,
    coupling_map: CouplingMap,
    qreg: QuantumRegister,
) -> Layout:
    """Greedy placement that directly minimizes sum of hardware distances.

    Places qubits one at a time, starting with the highest-interaction pair,
    greedily choosing the physical qubit that minimizes the routing distance
    to already-placed neighbors.
    """
    n_logical = circuit_adj.shape[0]
    n_physical = coupling_map.size()
    if n_logical == 0:
        return Layout()

    # Hardware shortest paths
    hw_adj = _coupling_map_to_adj(coupling_map)
    dist_matrix = shortest_path(hw_adj, directed=False, unweighted=True)

    # Circuit interaction edges sorted by weight (heaviest first)
    circ_coo = circuit_adj.tocoo()
    interactions = {}
    for u, v, w in zip(circ_coo.row, circ_coo.col, circ_coo.data):
        if u < v:
            interactions[(u, v)] = w

    # Sort by weight descending
    sorted_edges = sorted(interactions.items(), key=lambda x: -x[1])

    # Track assignments
    logical_to_physical = {}
    used_physical = set()

    # Find hardware center (lowest average distance)
    avg_dist = dist_matrix.mean(axis=1)
    center_order = np.argsort(avg_dist)

    if not sorted_edges:
        # No interactions — place in center order
        for i in range(n_logical):
            logical_to_physical[i] = int(center_order[i])
    else:
        # Place the first (heaviest) edge pair on the best adjacent hardware pair
        (first_a, first_b), _ = sorted_edges[0]

        # Find the best adjacent pair near the center
        best_pair_cost = float("inf")
        best_pa, best_pb = 0, 1
        hw_edges = coupling_map.get_edges()
        for pa, pb in hw_edges:
            pair_cost = avg_dist[pa] + avg_dist[pb]
            if pair_cost < best_pair_cost:
                best_pair_cost = pair_cost
                best_pa, best_pb = pa, pb

        logical_to_physical[first_a] = best_pa
        logical_to_physical[first_b] = best_pb
        used_physical.add(best_pa)
        used_physical.add(best_pb)

        # Place remaining qubits in priority order: next qubit = most
        # interactions with already-placed qubits (not index order).
        placed = set(logical_to_physical.keys())
        while len(placed) < n_logical:
            # Find next qubit with highest interaction to placed set
            best_next = -1
            best_priority = -1.0
            for i in range(n_logical):
                if i in placed:
                    continue
                priority = 0.0
                for j in placed:
                    priority += circuit_adj[i, j] + circuit_adj[j, i]
                # Tiebreak: prefer qubits with high total interaction degree
                if priority > best_priority or (priority == best_priority and best_next < 0):
                    best_priority = priority
                    best_next = i
            if best_next < 0:
                break

            # Find best physical qubit for this logical qubit
            best_cost = float("inf")
            best_phys = -1

            for p in range(n_physical):
                if p in used_physical:
                    continue
                cost = 0.0
                for j, pj in logical_to_physical.items():
                    w = circuit_adj[best_next, j] + circuit_adj[j, best_next]
                    if w > 0:
                        cost += w * dist_matrix[p, pj]
                # Tiebreak by centrality
                cost += 0.01 * avg_dist[p]
                if cost < best_cost:
                    best_cost = cost
                    best_phys = p

            if best_phys < 0:
                for p in range(n_physical):
                    if p not in used_physical:
                        best_phys = p
                        break

            logical_to_physical[best_next] = best_phys
            used_physical.add(best_phys)
            placed.add(best_next)

    layout_dict = {qreg[i]: logical_to_physical[i] for i in range(n_logical)}
    return Layout(layout_dict)


def sequential_band_match(
    circuit_adj: csr_matrix,
    coupling_map: CouplingMap,
    qreg: QuantumRegister,
) -> Layout:
    """Place qubits along hardware's longest geodesic, ordered by Fiedler vector.

    For QFT-like circuits with dense sequential interactions, this maps the
    natural linear ordering of qubits (from depth-weighted Fiedler vector)
    onto the hardware's longest chain. This minimizes routing distance for
    the highest-weight (earliest) interactions.
    """
    n_logical = circuit_adj.shape[0]
    n_physical = coupling_map.size()
    if n_logical == 0:
        return Layout()

    # Find Fiedler-based qubit ordering from the circuit interaction graph
    L_circ = build_standard_laplacian(circuit_adj, normalized=False)
    circ_order = _fiedler_ordering(L_circ, n_logical)

    # Find the hardware's longest geodesic path (diameter path)
    hw_adj = _coupling_map_to_adj(coupling_map)
    dist_matrix = shortest_path(hw_adj, directed=False, unweighted=True)
    hw_chain = _find_longest_chain(dist_matrix, coupling_map, n_logical)

    # Map Fiedler-ordered qubits onto the hardware chain
    layout_dict = {}
    for rank, log_idx in enumerate(circ_order):
        if rank < len(hw_chain):
            layout_dict[qreg[log_idx]] = hw_chain[rank]
        else:
            # Overflow: place on nearest unused physical qubit to chain end
            used = set(layout_dict.values())
            for p in range(n_physical):
                if p not in used:
                    layout_dict[qreg[log_idx]] = p
                    break

    return Layout(layout_dict)


def hierarchical_match(
    circuit_adj: csr_matrix,
    coupling_map: CouplingMap,
    qreg: QuantumRegister,
    params: Optional[HelixParams] = None,
    n_levels: int = 3,
) -> Layout:
    """Hierarchical multi-scale matching inspired by VHGT's 4-level wave hierarchy.

    Coarsens both circuit and hardware graphs at multiple scales, matches
    at the coarsest level first, then refines placement at each finer level.
    This captures global structure (which region of the hardware) at coarse
    scale and local optimization (exact qubit placement) at fine scale.

    Inspired by VHGT's hierarchy.py: 48→96→192→384 waves with bootstrap
    connections between levels. Here, each level doubles the resolution.
    """
    if params is None:
        params = HelixParams()

    n_logical = circuit_adj.shape[0]
    n_physical = coupling_map.size()
    if n_logical == 0:
        return Layout()

    hw_adj = _coupling_map_to_adj(coupling_map)
    dist_matrix = shortest_path(hw_adj, directed=False, unweighted=True)

    # --- Coarsen the circuit graph using Fiedler bisection ---
    # At each level, bisect the graph and treat each partition as a supernode
    # Level 0: full graph, Level k: 2^k partitions

    # Compute Fiedler vector for initial partitioning
    L_circ = build_standard_laplacian(circuit_adj, normalized=True)

    if n_logical < 6:
        # Too small for hierarchical approach — fall back to spectral
        return spectral_match(circuit_adj, hw_adj, qreg, coupling_map, params)

    # Build hierarchy of partitions via recursive Fiedler bisection
    def fiedler_bisect(nodes: list[int], adj_dense: np.ndarray) -> tuple[list[int], list[int]]:
        """Bisect a set of nodes using the Fiedler vector."""
        if len(nodes) < 2:
            return nodes, []
        # Extract subgraph
        sub_adj = adj_dense[np.ix_(nodes, nodes)]
        sub_L = np.diag(sub_adj.sum(axis=1)) - sub_adj
        # Add small regularization for disconnected subgraphs
        sub_L += np.eye(len(nodes)) * 1e-10
        evals, evecs = np.linalg.eigh(sub_L)
        # Fiedler vector = 2nd eigenvector
        fiedler_idx = 1 if len(evals) > 1 else 0
        fiedler = evecs[:, fiedler_idx]
        median = np.median(fiedler)
        part_a = [nodes[i] for i in range(len(nodes)) if fiedler[i] <= median]
        part_b = [nodes[i] for i in range(len(nodes)) if fiedler[i] > median]
        # Ensure both partitions are non-empty
        if not part_a:
            part_a = [part_b.pop(0)]
        if not part_b:
            part_b = [part_a.pop()]
        return part_a, part_b

    circ_dense = circuit_adj.toarray()
    hw_dense = hw_adj.toarray()

    # Build partition tree for circuit
    circ_partitions = [list(range(n_logical))]
    for level in range(min(n_levels, int(np.log2(max(n_logical, 2))))):
        new_partitions = []
        for part in circ_partitions:
            if len(part) >= 2:
                a, b = fiedler_bisect(part, circ_dense)
                new_partitions.extend([a, b])
            else:
                new_partitions.append(part)
        circ_partitions = new_partitions

    # Build partition tree for hardware
    hw_partitions = [list(range(n_physical))]
    for level in range(min(n_levels, int(np.log2(max(n_physical, 2))))):
        new_partitions = []
        for part in hw_partitions:
            if len(part) >= 2:
                a, b = fiedler_bisect(part, hw_dense)
                new_partitions.extend([a, b])
            else:
                new_partitions.append(part)
        hw_partitions = new_partitions

    # --- Match partitions at the coarsest level ---
    # Build supernode adjacency: interaction between circuit partitions
    n_cp = len(circ_partitions)
    n_hp = len(hw_partitions)

    # Map each node to its partition index
    circ_node_to_part = {}
    for pi, part in enumerate(circ_partitions):
        for node in part:
            circ_node_to_part[node] = pi

    hw_node_to_part = {}
    for pi, part in enumerate(hw_partitions):
        for node in part:
            hw_node_to_part[node] = pi

    # Compute centroids for each partition in spectral space
    hp = copy.copy(params)
    circ_coords = compute_spectral_coords(circuit_adj, hp)
    hw_coords = compute_spectral_coords(hw_adj, hp)

    # Centroid coordinates for each partition
    c_aligned, h_aligned = _align_coords(circ_coords, hw_coords)

    circ_centroids = np.zeros((n_cp, c_aligned.shape[1]))
    for pi, part in enumerate(circ_partitions):
        if part:
            circ_centroids[pi] = c_aligned[part].mean(axis=0)

    hw_centroids = np.zeros((n_hp, h_aligned.shape[1]))
    for pi, part in enumerate(hw_partitions):
        if part:
            hw_centroids[pi] = h_aligned[part].mean(axis=0)

    # Match partitions using cost matrix on centroids
    if n_cp > 0 and n_hp > 0:
        centroid_cost = np.zeros((n_cp, n_hp))
        for i in range(n_cp):
            for j in range(n_hp):
                centroid_cost[i, j] = np.sum((circ_centroids[i] - hw_centroids[j]) ** 2)

        part_row, part_col = linear_sum_assignment(centroid_cost)
        part_mapping = dict(zip(part_row, part_col))
    else:
        part_mapping = {}

    # --- Refine: within each matched partition pair, do fine-grained matching ---
    logical_to_physical = {}
    used_physical = set()

    for cp_idx, hp_idx in part_mapping.items():
        circ_nodes = circ_partitions[cp_idx]
        hw_nodes = hw_partitions[hp_idx]

        if not circ_nodes:
            continue

        # Available hardware nodes (exclude already used)
        avail_hw = [h for h in hw_nodes if h not in used_physical]
        if not avail_hw:
            continue

        # Build sub-cost matrix: spectral distance for nodes in this partition pair
        sub_cost = np.zeros((len(circ_nodes), len(avail_hw)))
        for i, cn in enumerate(circ_nodes):
            for j, hn in enumerate(avail_hw):
                sub_cost[i, j] = np.sum((c_aligned[cn] - h_aligned[hn]) ** 2)

        sub_row, sub_col = linear_sum_assignment(sub_cost)
        for i, j in zip(sub_row, sub_col):
            if i < len(circ_nodes) and j < len(avail_hw):
                logical_to_physical[circ_nodes[i]] = avail_hw[j]
                used_physical.add(avail_hw[j])

    # Handle any unmatched logical qubits
    for i in range(n_logical):
        if i not in logical_to_physical:
            for p in range(n_physical):
                if p not in used_physical:
                    logical_to_physical[i] = p
                    used_physical.add(p)
                    break

    layout_dict = {qreg[i]: logical_to_physical.get(i, i) for i in range(n_logical)}
    return Layout(layout_dict)


def _fiedler_ordering(L: csr_matrix, n: int) -> list[int]:
    """Get qubit ordering from Fiedler vector (2nd smallest eigenvector)."""
    if n < 2:
        return list(range(n))
    try:
        if n < 10:
            evals, evecs = np.linalg.eigh(L.toarray())
        else:
            evals, evecs = eigsh(L, k=min(3, n), which="SM", tol=1e-8, maxiter=2000)
        idx = np.argsort(evals)
        fiedler = evecs[:, idx[min(1, len(idx) - 1)]]
        return list(np.argsort(fiedler))
    except Exception:
        return list(range(n))


def _find_longest_chain(
    dist_matrix: np.ndarray,
    coupling_map: CouplingMap,
    target_length: int,
) -> list[int]:
    """Find a chain of target_length qubits along the hardware's diameter path."""
    n_physical = dist_matrix.shape[0]

    # Find the diameter endpoints
    diameter_pair = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    start, end = int(diameter_pair[0]), int(diameter_pair[1])

    # BFS to find shortest path from start to end
    from collections import deque
    edges = {}
    for u, v in coupling_map.get_edges():
        edges.setdefault(u, []).append(v)
        edges.setdefault(v, []).append(u)

    visited = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            break
        for neighbor in edges.get(node, []):
            if neighbor not in visited:
                visited[neighbor] = node
                queue.append(neighbor)

    # Reconstruct path
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = visited.get(node)
    path.reverse()

    # If path is shorter than target, extend with BFS from both ends
    if len(path) < target_length:
        used = set(path)
        # Extend from the end
        queue = deque(path)
        while len(path) < target_length and queue:
            node = queue.popleft()
            for neighbor in edges.get(node, []):
                if neighbor not in used and len(path) < target_length:
                    path.append(neighbor)
                    used.add(neighbor)
                    queue.append(neighbor)

    # If path is longer than target, take the central portion
    if len(path) > target_length:
        excess = len(path) - target_length
        start_idx = excess // 2
        path = path[start_idx:start_idx + target_length]

    return path


def _degree_sorted_match(
    circuit_adj: csr_matrix,
    coupling_map: CouplingMap,
    qreg: QuantumRegister,
) -> Layout:
    """Match by sorting: highest-interaction qubits → highest-degree hardware qubits."""
    n_logical = circuit_adj.shape[0]
    n_physical = coupling_map.size()
    if n_logical == 0:
        return Layout()

    # Circuit qubit degrees (interaction count)
    circ_degrees = np.array(circuit_adj.sum(axis=1)).flatten()
    circ_order = np.argsort(-circ_degrees)

    # Hardware qubit degrees
    hw_degrees = np.zeros(n_physical)
    for u, v in coupling_map.get_edges():
        hw_degrees[u] += 1
        hw_degrees[v] += 1
    hw_order = np.argsort(-hw_degrees)

    layout_dict = {}
    for rank, log_idx in enumerate(circ_order):
        if rank < n_physical:
            layout_dict[qreg[log_idx]] = int(hw_order[rank])

    return Layout(layout_dict)


def _coupling_map_to_adj(coupling_map: CouplingMap) -> csr_matrix:
    """Convert CouplingMap to a sparse adjacency matrix."""
    n = coupling_map.size()
    rows, cols, vals = [], [], []
    for u, v in coupling_map.get_edges():
        rows.extend([u, v])
        cols.extend([v, u])
        vals.extend([1.0, 1.0])
    return csr_matrix((vals, (rows, cols)), shape=(n, n))
