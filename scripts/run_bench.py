#!/usr/bin/env python3
"""CLI benchmark runner for ZynerjiQuantumCompiler."""

import argparse
import json
import sys
from pathlib import Path

from zynerji_qc.core.dual_helix import HelixParams
from zynerji_qc.bench import run_benchmark, format_results_table, default_benchmark_suite


def main():
    parser = argparse.ArgumentParser(description="Run ZynerjiQC benchmarks")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (JSON)")
    parser.add_argument("--omega", type=float, default=0.3)
    parser.add_argument("--c_log", type=float, default=1.0)
    parser.add_argument("--twist", type=float, default=0.33)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    params = HelixParams(
        omega=args.omega,
        c_log=args.c_log,
        twist_fraction=args.twist,
        k=args.k,
        alpha=args.alpha,
    )

    results = run_benchmark(params=params, verbose=not args.quiet)

    print("\n" + format_results_table(results))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "case": r.case_name,
                "method": r.method,
                "swaps": r.swap_count,
                "depth": r.depth,
                "time_ms": r.compile_time_ms,
                "n_qubits": r.n_qubits,
                "gates_orig": r.n_gates_original,
                "gates_compiled": r.n_gates_compiled,
            }
            for r in results
        ]
        out_path.write_text(json.dumps(data, indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
