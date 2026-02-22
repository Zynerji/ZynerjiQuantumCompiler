"""PassManager factory for the Helix compilation pipeline.

Provides:
- helix_pass_manager(): single-shot PassManager with Helix layout
- helix_transpile(): multi-strategy transpilation (spectral + greedy + random)
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import AnalysisPass, CouplingMap, PassManager, Target, Layout
from qiskit.transpiler.passes import (
    FullAncillaAllocation,
    EnlargeWithAncilla,
    ApplyLayout,
    SabreSwap,
    SabreLayout,
    Optimize1qGatesDecomposition,
    Unroll3qOrMore,
    BasisTranslator,
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary

from zynerji_qc.core.dual_helix import HelixParams
from zynerji_qc.passes.helix_layout import HelixLayout


def helix_pass_manager(
    coupling_map: CouplingMap,
    target: Optional[Target] = None,
    params: Optional[HelixParams] = None,
    basis_gates: Optional[list[str]] = None,
    depth_weighted: bool = False,
    error_weighted: bool = False,
    sabre_heuristic: str = "decay",
    optimize_1q: bool = True,
) -> PassManager:
    """Build a full PassManager with Helix layout + standard optimization."""
    if basis_gates is None:
        basis_gates = ["cx", "id", "rz", "sx", "x"]

    pm = PassManager()
    pm.append(Unroll3qOrMore())
    pm.append(BasisTranslator(SessionEquivalenceLibrary, basis_gates))
    pm.append(HelixLayout(
        coupling_map, target=target, params=params,
        depth_weighted=depth_weighted, error_weighted=error_weighted,
    ))
    pm.append(FullAncillaAllocation(coupling_map))
    pm.append(EnlargeWithAncilla())
    pm.append(ApplyLayout())
    pm.append(SabreSwap(coupling_map, heuristic=sabre_heuristic))
    if optimize_1q:
        pm.append(Optimize1qGatesDecomposition(basis=basis_gates))
    return pm


def _count_swaps(circuit: QuantumCircuit) -> int:
    """Count SWAP gates in a compiled circuit."""
    return sum(1 for inst in circuit.data if inst.operation.name == "swap")


def _route_with_layout(
    preprocessed: QuantumCircuit,
    layout: Layout,
    coupling_map: CouplingMap,
    basis_gates: list[str],
    heuristic: str = "decay",
    sabre_trials: int = 1,
    seed: int | None = None,
) -> QuantumCircuit:
    """Apply a fixed layout, route with SabreSwap, optimize 1Q gates."""
    pm = PassManager()
    pm.append(_SetLayout(layout))
    pm.append(FullAncillaAllocation(coupling_map))
    pm.append(EnlargeWithAncilla())
    pm.append(ApplyLayout())
    pm.append(SabreSwap(coupling_map, heuristic=heuristic, seed=seed, trials=sabre_trials))
    pm.append(Optimize1qGatesDecomposition(basis=basis_gates))
    return pm.run(preprocessed)


def _route_native_then_decompose(
    native_circuit: QuantumCircuit,
    layout: Layout,
    coupling_map: CouplingMap,
    basis_gates: list[str],
    sabre_trials: int = 1,
    seed: int | None = None,
) -> QuantumCircuit:
    """Route native gates (fewer 2Q ops), then decompose to basis.

    Native routing sees cp/cz/etc. as single 2Q gates instead of 2-3 CX gates,
    requiring fewer SWAPs. After routing, decompose non-basis gates while
    preserving SWAP gates.
    """
    pm = PassManager()
    pm.append(_SetLayout(layout))
    pm.append(FullAncillaAllocation(coupling_map))
    pm.append(EnlargeWithAncilla())
    pm.append(ApplyLayout())
    pm.append(SabreSwap(coupling_map, heuristic="decay", seed=seed, trials=sabre_trials))
    # Decompose to basis AFTER routing, preserving SWAP gates
    post_basis = list(basis_gates) + (["swap"] if "swap" not in basis_gates else [])
    pm.append(BasisTranslator(SessionEquivalenceLibrary, post_basis))
    pm.append(Optimize1qGatesDecomposition(basis=basis_gates))
    return pm.run(native_circuit)


def helix_transpile(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    target: Optional[Target] = None,
    params: Optional[HelixParams] = None,
    basis_gates: Optional[list[str]] = None,
    trials: int = 20,
    seed: int = 42,
) -> QuantumCircuit:
    """Multi-strategy transpilation: spectral + greedy + random perturbations.

    All candidates are routed identically (SabreSwap "decay"), so the
    only variable is layout quality. Returns the circuit with fewest SWAPs.
    """
    from zynerji_qc.core.interaction_graph import build_interaction_graph, build_signed_interaction_graph
    from zynerji_qc.core.spectral_match import (
        angular_spectral_match, distance_aware_match, greedy_distance_match,
        hall_spectral_match, hierarchical_match, qfactor_spectral_match,
        refine_layout_swaps, sequential_band_match, spectral_match,
    )
    from zynerji_qc.core.hardware_graph import build_hardware_graph

    if params is None:
        params = HelixParams()
    if basis_gates is None:
        basis_gates = ["cx", "id", "rz", "sx", "x"]

    rng = np.random.default_rng(seed)

    # Pre-processing: unroll 3+ qubit gates, then decompose to basis
    unroll_pm = PassManager()
    unroll_pm.append(Unroll3qOrMore())
    native_circuit = unroll_pm.run(circuit)
    native_dag = circuit_to_dag(native_circuit)

    decompose_pm = PassManager()
    decompose_pm.append(BasisTranslator(SessionEquivalenceLibrary, basis_gates))
    preprocessed = decompose_pm.run(native_circuit)
    dag = circuit_to_dag(preprocessed)

    n_logical = dag.num_qubits()
    qregs = list(dag.qregs.values())
    if not qregs or n_logical == 0:
        return preprocessed
    qreg = qregs[0]
    n_physical = coupling_map.size()

    # Build interaction graphs from NATIVE circuit (cleaner spectral signal).
    # Native cp/cz/etc. = 1 gate per interaction vs CX-decomposed = 2-3.
    # Gives better Fiedler ordering and spectral structure for layout decisions.
    native_adj = build_interaction_graph(native_dag)
    native_adj_dw = build_interaction_graph(native_dag, depth_weighted=True)
    # Also keep decomposed interaction graphs for routing-aware refinement
    circuit_adj = build_interaction_graph(dag)
    circuit_adj_dw = build_interaction_graph(dag, depth_weighted=True)

    # Classify circuit density for adaptive routing
    max_possible = n_logical * (n_logical - 1) / 2
    nnz_pairs = native_adj.nnz / 2
    density = nnz_pairs / max(1, max_possible)
    is_dense = density > 0.5  # QFT-like complete interaction

    candidate_layouts = []

    # === Layout generation strategies ===

    # Strategy 1: Greedy distance-aware placement (native + depth-weighted)
    for name, adj in [("greedy", native_adj), ("greedy_dw", native_adj_dw),
                      ("greedy_cx", circuit_adj)]:
        try:
            layout = greedy_distance_match(adj, coupling_map, qreg)
            candidate_layouts.append((name, layout))
        except Exception:
            pass

    # Strategy 1b: Distance-aware LAP matching (directly optimizes routing cost)
    for name, adj in [("dist_aware", native_adj), ("dist_aware_dw", native_adj_dw)]:
        try:
            layout = distance_aware_match(adj, coupling_map, qreg)
            candidate_layouts.append((name, layout))
        except Exception:
            pass

    # Strategy 2: Spectral layouts with parameter diversity
    for mode_name, use_helix, alpha, omega in [
        ("helix", True, 3.0, 0.3), ("spectral", False, 3.0, 0.3),
        ("helix_a1", True, 1.0, 0.3), ("spectral_a1", False, 1.0, 0.3),
        ("helix_a5", True, 5.0, 0.3), ("spectral_a5", False, 5.0, 0.3),
        ("helix_w1", True, 3.0, 0.1), ("helix_w5", True, 3.0, 0.5),
        ("helix_w1a1", True, 1.0, 0.1),
    ]:
        try:
            hp = copy.copy(params)
            hp.use_helix = use_helix
            hp.alpha = alpha
            hp.omega = omega
            lp = HelixLayout(coupling_map, target=target, params=hp)
            lp.run(native_dag)
            if "layout" in lp.property_set:
                candidate_layouts.append((mode_name, lp.property_set["layout"]))
        except Exception:
            pass

    # Strategy 2b: Depth-weighted spectral matching (breaks K_n degeneracy for QFT)
    try:
        hw_adj = build_hardware_graph(coupling_map, target=target)
        for mode_name, adj, use_helix, alpha, k in [
            ("spectral_dw", native_adj_dw, False, 3.0, 8),
            ("helix_dw", native_adj_dw, True, 3.0, 8),
            ("spectral_cx_dw", circuit_adj_dw, False, 3.0, 8),
            ("spectral_dw_a1", native_adj_dw, False, 1.0, 8),
            ("helix_dw_a1", native_adj_dw, True, 1.0, 8),
            ("spectral_dw_k4", native_adj_dw, False, 3.0, 4),
            ("helix_dw_k4", native_adj_dw, True, 3.0, 4),
            ("spectral_dw_k16", native_adj_dw, False, 3.0, 16),
        ]:
            hp = copy.copy(params)
            hp.use_helix = use_helix
            hp.alpha = alpha
            hp.k = k
            layout = spectral_match(adj, hw_adj, qreg, coupling_map, hp)
            candidate_layouts.append((mode_name, layout))
    except Exception:
        pass

    # Strategy 3: Hall spectral placement (Fiedler x, 3rd eigvec y)
    for name, adj, norm in [
        ("hall", native_adj, True),
        ("hall_dw", native_adj_dw, True),
        ("hall_unnorm", native_adj, False),
        ("hall_cx_dw", circuit_adj_dw, True),
    ]:
        try:
            layout = hall_spectral_match(adj, coupling_map, qreg, normalized=norm)
            candidate_layouts.append((name, layout))
        except Exception:
            pass

    # Strategy 4: Sequential band placement (QFT-optimized)
    for name, adj in [("seqband", native_adj), ("seqband_dw", native_adj_dw)]:
        try:
            layout = sequential_band_match(adj, coupling_map, qreg)
            candidate_layouts.append((name, layout))
        except Exception:
            pass

    # === VHGT-inspired strategies ===

    # Strategy V1: Angular (phase) matching — cosine similarity instead of L2
    # From VHGT: phase alignment captures structural role similarity
    try:
        hw_adj_v = build_hardware_graph(coupling_map, target=target)
        for mode_name, adj, use_helix, alpha, ang_w in [
            ("angular", native_adj, False, 3.0, 1.0),
            ("angular_h", native_adj, True, 3.0, 1.0),
            ("hybrid_50", native_adj, False, 3.0, 0.5),
            ("hybrid_h50", native_adj, True, 3.0, 0.5),
            ("angular_dw", native_adj_dw, False, 3.0, 1.0),
            ("hybrid_dw50", native_adj_dw, False, 3.0, 0.5),
        ]:
            hp = copy.copy(params)
            hp.use_helix = use_helix
            hp.alpha = alpha
            layout = angular_spectral_match(adj, hw_adj_v, qreg, coupling_map, hp, ang_w)
            candidate_layouts.append((mode_name, layout))
    except Exception:
        pass

    # Strategy V2: Ternary signed adjacency (attraction + repulsion)
    # From VHGT: {-1, 0, +1} encoding. Competing qubits get pushed apart.
    try:
        signed_adj = build_signed_interaction_graph(native_dag)
        for name, adj, use_helix in [
            ("signed", signed_adj, False),
            ("signed_h", signed_adj, True),
        ]:
            hp = copy.copy(params)
            hp.use_helix = use_helix
            layout = spectral_match(adj, hw_adj_v, qreg, coupling_map, hp)
            candidate_layouts.append((name, layout))
        # Also try signed + angular
        hp = copy.copy(params)
        hp.use_helix = False
        layout = angular_spectral_match(signed_adj, hw_adj_v, qreg, coupling_map, hp, 1.0)
        candidate_layouts.append(("signed_angular", layout))
    except Exception:
        pass

    # Strategy V3: Golden ratio eigenvector selection
    # From VHGT: anti-periodic golden spacing avoids harmonic redundancy
    try:
        from zynerji_qc.core.dual_helix import compute_spectral_coords as _csc
        from zynerji_qc.core.spectral_match import build_cost_matrix, build_angular_cost_matrix
        for mode_name, adj, use_helix, cost_fn in [
            ("golden", native_adj, False, "l2"),
            ("golden_h", native_adj, True, "l2"),
            ("golden_ang", native_adj, False, "angular"),
            ("golden_dw", native_adj_dw, False, "l2"),
        ]:
            hp = copy.copy(params)
            hp.use_helix = use_helix
            circ_coords = _csc(adj, hp, golden_selection=True)
            hw_coords = _csc(hw_adj_v, hp, golden_selection=True)
            if cost_fn == "angular":
                cost = build_angular_cost_matrix(circ_coords, hw_coords)
            else:
                cost = build_cost_matrix(circ_coords, hw_coords)
            from scipy.optimize import linear_sum_assignment as _lap
            row_ind, col_ind = _lap(cost)
            layout_dict = {}
            for log_idx, phys_idx in zip(row_ind, col_ind):
                if log_idx < n_logical:
                    layout_dict[qreg[log_idx]] = phys_idx
            candidate_layouts.append((mode_name, Layout(layout_dict)))
    except Exception:
        pass

    # Strategy V3b: Q-factor amplification at structural antinodes
    # From VHGT: amplify matches at structurally aligned positions
    try:
        for mode_name, adj, use_helix, q_amp, thresh in [
            ("qfact", native_adj, False, 1.5, 0.85),
            ("qfact_h", native_adj, True, 1.5, 0.85),
            ("qfact_2x", native_adj, False, 2.0, 0.8),
            ("qfact_dw", native_adj_dw, False, 1.5, 0.85),
        ]:
            hp = copy.copy(params)
            hp.use_helix = use_helix
            layout = qfactor_spectral_match(
                adj, hw_adj_v, qreg, coupling_map, hp,
                q_amplification=q_amp, alignment_threshold=thresh,
            )
            candidate_layouts.append((mode_name, layout))
    except Exception:
        pass

    # Strategy V4: Hierarchical multi-scale matching
    # From VHGT: 4-level wave hierarchy → coarsen, match, refine
    for name, adj, n_lvl in [
        ("hier2", native_adj, 2),
        ("hier3", native_adj, 3),
        ("hier_dw", native_adj_dw, 3),
    ]:
        try:
            layout = hierarchical_match(adj, coupling_map, qreg, params, n_levels=n_lvl)
            candidate_layouts.append((name, layout))
        except Exception:
            pass

    # Strategy 5: SABRE layout (iterative forward/backward, layout only)
    # Run on BOTH native and decomposed DAGs — different gate counts give different layouts
    sabre_configs = [
        (40, 4, seed),
        (20, 8, seed + 1),
        (40, 8, seed + 2),
        (60, 4, seed + 3),
        (20, 12, seed + 4),
    ]
    if is_dense:
        sabre_configs.extend([
            (80, 4, seed + 5),
            (40, 12, seed + 6),
            (100, 4, seed + 7),
        ])
    for sabre_idx, (s_trials, s_iters, s_seed) in enumerate(sabre_configs):
        for dag_label, dag_src in [("sabre", dag), ("sabre_nat", native_dag)]:
            try:
                dag_copy = copy.deepcopy(dag_src)
                sl = SabreLayout(
                    coupling_map, seed=s_seed, skip_routing=True,
                    layout_trials=s_trials, max_iterations=s_iters,
                )
                sl.run(dag_copy)
                if "layout" in sl.property_set:
                    candidate_layouts.append(
                        (f"{dag_label}_{sabre_idx}", sl.property_set["layout"])
                    )
            except Exception:
                pass

    # Refine all intelligent layouts with pairwise swap local search
    n_pre_refine = len(candidate_layouts)
    try:
        refined = []
        for name, base_layout in list(candidate_layouts):
            try:
                rl = refine_layout_swaps(
                    base_layout, circuit_adj, coupling_map, qreg, max_rounds=5,
                )
                refined.append((f"{name}_ref", rl))
            except Exception:
                continue
        candidate_layouts.extend(refined)
    except Exception:
        pass

    # Add perturbations of top intelligent layouts
    n_intelligent = len(candidate_layouts)
    n_base = min(n_pre_refine, 8)
    for base_name, base_layout in list(candidate_layouts[:n_base]):
        phys_map = []
        for i in range(n_logical):
            try:
                phys_map.append(base_layout[qreg[i]])
            except (KeyError, IndexError):
                phys_map.append(i)
        for t in range(5):
            perm = list(phys_map)
            n_sw = rng.integers(1, min(4, max(2, n_logical // 2)))
            for _ in range(n_sw):
                if n_logical >= 2:
                    i, j = rng.choice(n_logical, size=2, replace=False)
                    perm[i], perm[j] = perm[j], perm[i]
            layout_dict = {qreg[i]: perm[i] for i in range(n_logical)}
            candidate_layouts.append((f"{base_name}_p{t}", Layout(layout_dict)))

    # Fill remaining budget with random layouts
    for t in range(max(0, trials - len(candidate_layouts))):
        phys_qubits = rng.permutation(n_physical)[:n_logical].tolist()
        layout_dict = {qreg[i]: phys_qubits[i] for i in range(n_logical)}
        candidate_layouts.append((f"random_{t}", Layout(layout_dict)))

    # === Strategy 0: Multi-seed Qiskit as ensemble baseline ===
    # Guarantees we're never worse than Qiskit; multiple seeds explore diversity
    from qiskit import transpile as qiskit_transpile
    best_circuit = None
    best_swaps = float("inf")
    # opt_level=3 and opt_level=2 produce different SABRE configurations
    for opt_lvl in [3, 2]:
        for qiskit_seed in range(seed, seed + 8):
            try:
                qr = qiskit_transpile(
                    circuit, coupling_map=coupling_map, optimization_level=opt_lvl,
                    seed_transpiler=qiskit_seed,
                )
                qs = _count_swaps(qr)
                if qs < best_swaps:
                    best_swaps = qs
                    best_circuit = qr
            except Exception:
                pass

    # === Deduplicate layouts before routing ===
    seen_maps = set()
    unique_layouts = []
    for name, layout in candidate_layouts:
        try:
            key = tuple(layout[qreg[i]] for i in range(n_logical))
        except (KeyError, IndexError):
            key = None
        if key is not None and key in seen_maps:
            continue
        if key is not None:
            seen_maps.add(key)
        unique_layouts.append((name, layout))
    candidate_layouts = unique_layouts

    # === Stage 1: Evaluate all candidates ===
    eval_trials = 4 if is_dense else 2
    top_k = []  # (swap_count, name, layout)
    for name, layout in candidate_layouts:
        try:
            compiled = _route_with_layout(
                preprocessed, layout, coupling_map, basis_gates,
                sabre_trials=eval_trials,
            )
            swaps = _count_swaps(compiled)
            if swaps < best_swaps:
                best_swaps = swaps
                best_circuit = compiled
            top_k.append((swaps, name, layout))
        except Exception:
            continue

    # === Stage 2: Deep multi-seed routing on top layouts ===
    if top_k and n_logical >= 4:
        top_k.sort(key=lambda x: x[0])
        route_trials = 8 if is_dense else 4
        n_top = min(10, len(top_k))
        for _, _, top_layout in top_k[:n_top]:
            for route_seed in range(seed, seed + 10):
                try:
                    compiled = _route_with_layout(
                        preprocessed, top_layout, coupling_map, basis_gates,
                        sabre_trials=route_trials, seed=route_seed,
                    )
                    swaps = _count_swaps(compiled)
                    if swaps < best_swaps:
                        best_swaps = swaps
                        best_circuit = compiled
                except Exception:
                    continue

    # === Stage 2b: Temporal loop refinement on top candidates ===
    if top_k and n_logical >= 4:
        n_refine = min(5, len(top_k))
        for swaps, name, layout in top_k[:n_refine]:
            try:
                refined_circuit, refined_swaps = _temporal_loop_refine(
                    preprocessed, layout, coupling_map, basis_gates,
                    qreg, n_logical, max_iterations=4, seed=seed,
                )
                if refined_swaps < best_swaps:
                    best_swaps = refined_swaps
                    best_circuit = refined_circuit
            except Exception:
                continue

    # === Stage 3: Qiskit full pipeline with our best layouts ===
    # Qiskit's routing+optimization with our spectral layout = best of both worlds
    if top_k and n_logical >= 4:
        n_top_qiskit = min(8, len(top_k))
        for _, _, top_layout in top_k[:n_top_qiskit]:
            try:
                layout_dict = {qreg[i]: top_layout[qreg[i]]
                               for i in range(n_logical)}
            except (KeyError, IndexError):
                continue
            for opt_lvl in [3, 2]:
                for q_seed in range(seed, seed + 5):
                    try:
                        compiled = qiskit_transpile(
                            circuit, coupling_map=coupling_map,
                            optimization_level=opt_lvl, initial_layout=layout_dict,
                            seed_transpiler=q_seed,
                        )
                        swaps = _count_swaps(compiled)
                        if swaps < best_swaps:
                            best_swaps = swaps
                            best_circuit = compiled
                    except Exception:
                        continue

    if best_circuit is None:
        from qiskit import transpile
        return transpile(circuit, coupling_map=coupling_map, optimization_level=3,
                         seed_transpiler=seed)

    return best_circuit


def _temporal_loop_refine(
    preprocessed: QuantumCircuit,
    initial_layout: Layout,
    coupling_map: CouplingMap,
    basis_gates: list[str],
    qreg,
    n_logical: int,
    max_iterations: int = 4,
    seed: int = 42,
) -> tuple[QuantumCircuit, int]:
    """Temporal loop layout refinement (forward/backward convergence).

    Implements the retrocausal blender pattern from HHmL/ZynerjiTrader:
    Forward pass routes the circuit, backward pass discovers better layouts
    by routing the reversed gate order. Iterate until convergence.

    Returns (best_circuit, best_swap_count).
    """
    # Forward pass with initial layout
    best_circuit = _route_with_layout(preprocessed, initial_layout, coupling_map, basis_gates)
    best_swaps = _count_swaps(best_circuit)
    current_layout = initial_layout

    # Build reversed circuit (same gates, reversed order)
    reversed_qc = preprocessed.copy()
    reversed_qc.data = list(reversed(reversed_qc.data))

    for iteration in range(max_iterations):
        # Backward pass: route reversed circuit with current layout
        try:
            pm_back = PassManager()
            pm_back.append(_SetLayout(current_layout))
            pm_back.append(FullAncillaAllocation(coupling_map))
            pm_back.append(EnlargeWithAncilla())
            pm_back.append(ApplyLayout())
            pm_back.append(SabreSwap(coupling_map, heuristic="decay", seed=seed + iteration))
            routed_back = pm_back.run(reversed_qc)

            # Extract final layout from the backward pass
            final_layout = pm_back.property_set.get("final_layout")
            if final_layout is None:
                break

            # Compose: initial_layout o final_layout^{-1} gives new starting layout
            # The final_layout maps virtual->physical after routing
            # We want to use this permutation as our new initial layout
            new_layout_dict = {}
            for i in range(n_logical):
                vbit = qreg[i]
                try:
                    # The final_layout tells us where each virtual qubit ended up
                    # Use this as the new starting position
                    phys = final_layout[preprocessed.qubits[i]]
                    new_layout_dict[vbit] = phys
                except (KeyError, IndexError):
                    # Fall back to current layout
                    try:
                        new_layout_dict[vbit] = current_layout[vbit]
                    except (KeyError, IndexError):
                        new_layout_dict[vbit] = i

            new_layout = Layout(new_layout_dict)

            # Forward pass with the new layout
            compiled = _route_with_layout(preprocessed, new_layout, coupling_map, basis_gates)
            swaps = _count_swaps(compiled)

            if swaps < best_swaps:
                best_swaps = swaps
                best_circuit = compiled
                current_layout = new_layout
            else:
                # Convergence: no improvement, stop iterating
                break

        except Exception:
            break

    return best_circuit, best_swaps


class _SetLayout(AnalysisPass):
    """AnalysisPass that injects a pre-computed layout."""

    def __init__(self, layout: Layout):
        super().__init__()
        self._layout = layout

    def run(self, dag):
        self.property_set["layout"] = self._layout
