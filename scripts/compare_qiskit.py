#!/usr/bin/env python3
"""Head-to-head comparison: Helix portfolio vs Qiskit opt_level=3 vs random-layout baseline.

The key comparison is:
- helix: Our spectral/greedy layouts + random perturbations, routed with SabreSwap
- random_20: 20 random layouts, same SabreSwap routing (shows layout quality)
- qiskit_opt3: Qiskit's full SabreLayout + routing pipeline
"""

import argparse
import time
from collections import defaultdict

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, PassManager, Layout
from qiskit.transpiler.passes import (
    FullAncillaAllocation, EnlargeWithAncilla, ApplyLayout,
    SabreSwap, Optimize1qGatesDecomposition, Unroll3qOrMore, BasisTranslator,
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.converters import circuit_to_dag

from zynerji_qc.core.dual_helix import HelixParams
from zynerji_qc.bench import default_benchmark_suite, count_swaps
from zynerji_qc.pipeline import helix_transpile, _SetLayout, _route_with_layout


BASIS = ["cx", "id", "rz", "sx", "x"]


def random_baseline(circuit, coupling_map, n_trials=20, seed=42):
    """Run n_trials random layouts through same routing as Helix."""
    rng = np.random.default_rng(seed)
    pre_pm = PassManager()
    pre_pm.append(Unroll3qOrMore())
    pre_pm.append(BasisTranslator(SessionEquivalenceLibrary, BASIS))
    preprocessed = pre_pm.run(circuit)
    dag = circuit_to_dag(preprocessed)
    n_logical = dag.num_qubits()
    n_physical = coupling_map.size()
    qregs = list(dag.qregs.values())
    if not qregs or n_logical == 0:
        return preprocessed
    qreg = qregs[0]

    best_circuit = None
    best_swaps = float("inf")

    for _ in range(n_trials):
        phys = rng.permutation(n_physical)[:n_logical].tolist()
        layout = Layout({qreg[i]: phys[i] for i in range(n_logical)})
        try:
            compiled = _route_with_layout(preprocessed, layout, coupling_map, BASIS)
            swaps = count_swaps(compiled)
            if swaps < best_swaps:
                best_swaps = swaps
                best_circuit = compiled
        except Exception:
            continue

    return best_circuit


def main():
    parser = argparse.ArgumentParser(description="Compare Helix vs Qiskit vs Random")
    parser.add_argument("--omega", type=float, default=0.3)
    parser.add_argument("--c_log", type=float, default=1.0)
    parser.add_argument("--twist", type=float, default=0.33)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    params = HelixParams(
        omega=args.omega, c_log=args.c_log, twist_fraction=args.twist,
        k=args.k, alpha=args.alpha,
    )

    cases = default_benchmark_suite()
    by_case = {}

    for case in cases:
        circuit = case.circuit_fn(*case.circuit_args)
        n_q = circuit.num_qubits
        n_g = len(circuit.data)
        print(f"\n--- {case.name} ({n_q}q, {n_g} gates) ---")
        results = {}

        # Helix portfolio
        try:
            t0 = time.perf_counter()
            compiled = helix_transpile(circuit, case.coupling_map, params=params,
                                       trials=args.trials)
            t1 = time.perf_counter()
            s = count_swaps(compiled)
            results["helix"] = (s, compiled.depth(), (t1-t0)*1000)
            print(f"  Helix:    {s:>4} SWAPs, depth {compiled.depth():>4}, {(t1-t0)*1000:.0f}ms")
        except Exception as e:
            print(f"  Helix:    FAILED ({e})")

        # Random baseline (same routing, random layouts)
        try:
            t0 = time.perf_counter()
            compiled = random_baseline(circuit, case.coupling_map, n_trials=args.trials)
            t1 = time.perf_counter()
            if compiled:
                s = count_swaps(compiled)
                results["random"] = (s, compiled.depth(), (t1-t0)*1000)
                print(f"  Random:   {s:>4} SWAPs, depth {compiled.depth():>4}, {(t1-t0)*1000:.0f}ms")
        except Exception as e:
            print(f"  Random:   FAILED ({e})")

        # Qiskit opt=3
        try:
            t0 = time.perf_counter()
            compiled = transpile(circuit, coupling_map=case.coupling_map,
                                 optimization_level=3, seed_transpiler=42)
            t1 = time.perf_counter()
            s = count_swaps(compiled)
            results["qiskit3"] = (s, compiled.depth(), (t1-t0)*1000)
            print(f"  Qiskit3:  {s:>4} SWAPs, depth {compiled.depth():>4}, {(t1-t0)*1000:.0f}ms")
        except Exception as e:
            print(f"  Qiskit3:  FAILED ({e})")

        by_case[case.name] = results

    # Summary table
    print("\n\n=== Results Summary ===\n")
    print(f"{'Case':<30} {'Helix':>7} {'Random':>7} {'Qiskit3':>7} {'H-Q':>6} {'H-R':>6}")
    print("-" * 70)

    total = {"helix": 0, "random": 0, "qiskit3": 0}
    count = 0

    for name in sorted(by_case.keys()):
        r = by_case[name]
        h = r.get("helix", (None,))[0]
        rand = r.get("random", (None,))[0]
        q = r.get("qiskit3", (None,))[0]

        h_str = f"{h:>7}" if h is not None else "   FAIL"
        r_str = f"{rand:>7}" if rand is not None else "   FAIL"
        q_str = f"{q:>7}" if q is not None else "   FAIL"
        hq = f"{h - q:>+6}" if h is not None and q is not None else "     -"
        hr = f"{h - rand:>+6}" if h is not None and rand is not None else "     -"

        print(f"{name:<30} {h_str} {r_str} {q_str} {hq} {hr}")

        if h is not None: total["helix"] += h
        if rand is not None: total["random"] += rand
        if q is not None: total["qiskit3"] += q
        if h is not None and q is not None: count += 1

    print("-" * 70)
    h_t, r_t, q_t = total["helix"], total["random"], total["qiskit3"]
    print(f"{'TOTAL':<30} {h_t:>7} {r_t:>7} {q_t:>7} {h_t-q_t:>+6} {h_t-r_t:>+6}")

    if q_t > 0:
        print(f"\nHelix vs Qiskit3: {(q_t-h_t)/q_t*100:>+.1f}% (positive = Helix wins)")
    if r_t > 0:
        print(f"Helix vs Random:  {(r_t-h_t)/r_t*100:>+.1f}% (positive = Helix wins)")
    if q_t > 0 and r_t > 0:
        print(f"Random vs Qiskit: {(q_t-r_t)/q_t*100:>+.1f}%")


if __name__ == "__main__":
    main()
