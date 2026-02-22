#!/usr/bin/env python3
"""Large QFT benchmarks: QFT-64 and QFT-128 on grid and heavy-hex topologies."""

import time
import numpy as np

from qiskit import transpile
from qiskit.transpiler import CouplingMap

from zynerji_qc.bench import make_qft, make_heavy_hex, make_grid, count_swaps
from zynerji_qc.pipeline import helix_transpile
from zynerji_qc.core.dual_helix import HelixParams


def run_case(name, circuit, coupling_map, params):
    n_q = circuit.num_qubits
    n_g = len(circuit.data)
    print(f"\n{'='*70}")
    print(f"--- {name} ({n_q}q, {n_g} gates, topology {coupling_map.size()}q) ---")
    print(f"{'='*70}")

    results = {}

    # Helix
    print("  Running Helix...", flush=True)
    try:
        t0 = time.perf_counter()
        compiled = helix_transpile(circuit, coupling_map, params=params, trials=20)
        t1 = time.perf_counter()
        s = count_swaps(compiled)
        results["helix"] = s
        print(f"  Helix:   {s:>5} SWAPs, depth {compiled.depth():>5}, "
              f"{(t1-t0):.1f}s")
    except Exception as e:
        print(f"  Helix:   FAILED ({e})")

    # Qiskit opt=3 (best of 5 seeds for fair comparison)
    print("  Running Qiskit opt=3 (best of 5 seeds)...", flush=True)
    try:
        best_q = float("inf")
        best_qr = None
        t0 = time.perf_counter()
        for seed in [42, 43, 44, 45, 46]:
            qr = transpile(circuit, coupling_map=coupling_map,
                           optimization_level=3, seed_transpiler=seed)
            qs = count_swaps(qr)
            if qs < best_q:
                best_q = qs
                best_qr = qr
        t1 = time.perf_counter()
        results["qiskit3"] = best_q
        print(f"  Qiskit3: {best_q:>5} SWAPs, depth {best_qr.depth():>5}, "
              f"{(t1-t0):.1f}s")
    except Exception as e:
        print(f"  Qiskit3: FAILED ({e})")

    # Summary
    if "helix" in results and "qiskit3" in results:
        h, q = results["helix"], results["qiskit3"]
        diff = h - q
        pct = (q - h) / q * 100 if q > 0 else 0
        winner = "HELIX WINS" if h < q else "TIED" if h == q else "Qiskit wins"
        print(f"\n  >> {winner}: Helix {h} vs Qiskit {q} ({diff:+d} SWAPs, {pct:+.1f}%)")

    return results


def main():
    params = HelixParams()

    all_results = {}

    # QFT-64 on 12x12 grid (144 physical qubits)
    print("\n\nBuilding QFT-64...")
    qft64 = make_qft(64)
    grid_12 = make_grid(12, 12)
    all_results["qft_64_grid_12x12"] = run_case(
        "qft_64_grid_12x12", qft64, grid_12, params)

    # QFT-64 on heavy-hex (IBM Eagle-like, requires odd distance)
    hhex_65 = make_heavy_hex(5)  # distance=5 -> typically 65 qubits
    print(f"\nHeavy-hex d=5: {hhex_65.size()} qubits")
    if hhex_65.size() >= 64:
        all_results["qft_64_heavyhex"] = run_case(
            f"qft_64_heavyhex_{hhex_65.size()}", qft64, hhex_65, params)
    else:
        hhex_big = make_heavy_hex(7)  # d must be odd
        print(f"Heavy-hex d=7: {hhex_big.size()} qubits")
        if hhex_big.size() >= 64:
            all_results["qft_64_heavyhex"] = run_case(
                f"qft_64_heavyhex_{hhex_big.size()}", qft64, hhex_big, params)

    # QFT-128 on 16x16 grid (256 physical qubits)
    print("\n\nBuilding QFT-128...")
    qft128 = make_qft(128)
    grid_16 = make_grid(16, 16)
    all_results["qft_128_grid_16x16"] = run_case(
        "qft_128_grid_16x16", qft128, grid_16, params)

    # QFT-128 on heavy-hex (odd distances only)
    hhex_big = make_heavy_hex(9)  # distance=9 -> 193 qubits
    print(f"\nHeavy-hex d=9: {hhex_big.size()} qubits")
    if hhex_big.size() >= 128:
        all_results["qft_128_heavyhex"] = run_case(
            f"qft_128_heavyhex_{hhex_big.size()}", qft128, hhex_big, params)

    # Final summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    total_h = 0
    total_q = 0
    for name, r in all_results.items():
        h = r.get("helix", "FAIL")
        q = r.get("qiskit3", "FAIL")
        if isinstance(h, int) and isinstance(q, int):
            pct = (q - h) / q * 100
            print(f"  {name:<35} Helix: {h:>5}  Qiskit3: {q:>5}  "
                  f"({pct:+.1f}%)")
            total_h += h
            total_q += q
        else:
            print(f"  {name:<35} Helix: {h}  Qiskit3: {q}")

    if total_q > 0:
        total_pct = (total_q - total_h) / total_q * 100
        print(f"\n  {'TOTAL':<35} Helix: {total_h:>5}  Qiskit3: {total_q:>5}  "
              f"({total_pct:+.1f}%)")


if __name__ == "__main__":
    main()
