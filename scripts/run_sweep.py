#!/usr/bin/env python3
"""CLI parameter sweep runner for ZynerjiQuantumCompiler."""

import argparse
import json
from pathlib import Path

from zynerji_qc.sweep import run_sweep, refine_sweep, SweepConfig
from zynerji_qc.core.dual_helix import HelixParams


def main():
    parser = argparse.ArgumentParser(description="Run ZynerjiQC parameter sweep")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (JSON)")
    parser.add_argument("--refine", action="store_true",
                        help="Run fine-grained refinement after coarse sweep")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Coarse sweep
    results = run_sweep(n_workers=args.workers, verbose=not args.quiet)

    if args.refine and results:
        print("\n=== Refinement sweep ===")
        best = results[0].params
        refined = refine_sweep(best, n_workers=args.workers, verbose=not args.quiet)
        results = refined

    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "omega": r.params.omega,
                "c_log": r.params.c_log,
                "twist_fraction": r.params.twist_fraction,
                "k": r.params.k,
                "alpha": r.params.alpha,
                "total_swaps": r.total_swaps,
                "total_depth": r.total_depth,
                "total_time_ms": r.total_time_ms,
                "per_case": r.per_case,
            }
            for r in results[:20]
        ]
        out_path.write_text(json.dumps(data, indent=2))
        print(f"\nResults saved to {out_path}")

    if results:
        best = results[0]
        p = best.params
        print(f"\nBest params: omega={p.omega}, c_log={p.c_log}, "
              f"twist={p.twist_fraction}, k={p.k}, alpha={p.alpha}")
        print(f"Total SWAPs: {best.total_swaps}")


if __name__ == "__main__":
    main()
