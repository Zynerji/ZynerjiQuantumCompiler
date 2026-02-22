"""Parameter sweep and auto-tuner for Helix compilation.

Grid searches over (omega, c_log, twist_fraction, k, alpha) to minimize
total SWAP count across a benchmark suite. Uses multiprocessing for
parallel evaluation.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from zynerji_qc.core.dual_helix import HelixParams
from zynerji_qc.pipeline import helix_pass_manager
from zynerji_qc.bench import (
    BenchmarkCase,
    default_benchmark_suite,
    count_swaps,
)


@dataclass
class SweepResult:
    """Result of evaluating one parameter configuration."""
    params: HelixParams
    total_swaps: int
    total_depth: int
    total_time_ms: float
    per_case: dict  # case_name -> swap_count


@dataclass
class SweepConfig:
    """Configuration for the parameter sweep grid."""
    omega_range: list[float] = None
    c_log_range: list[float] = None
    twist_fraction_range: list[float] = None
    k_range: list[int] = None
    alpha_range: list[float] = None

    def __post_init__(self):
        if self.omega_range is None:
            self.omega_range = [0.1, 0.2, 0.3, 0.5, 0.8]
        if self.c_log_range is None:
            self.c_log_range = [0.5, 1.0, 2.0]
        if self.twist_fraction_range is None:
            self.twist_fraction_range = [0.2, 0.33, 0.5]
        if self.k_range is None:
            self.k_range = [4, 8, 12]
        if self.alpha_range is None:
            self.alpha_range = [1.0, 3.0, 5.0]

    def grid(self) -> list[HelixParams]:
        """Generate all parameter combinations."""
        combos = itertools.product(
            self.omega_range,
            self.c_log_range,
            self.twist_fraction_range,
            self.k_range,
            self.alpha_range,
        )
        return [
            HelixParams(omega=o, c_log=c, twist_fraction=t, k=k, alpha=a)
            for o, c, t, k, a in combos
        ]


def _evaluate_params(args: tuple) -> SweepResult:
    """Evaluate a single parameter configuration across all benchmark cases.

    This function is designed to be called via multiprocessing.Pool.map().
    """
    params, cases_data = args

    total_swaps = 0
    total_depth = 0
    total_time = 0.0
    per_case = {}

    for name, circuit_data, coupling_edges, n_hw_qubits in cases_data:
        try:
            circuit = QuantumCircuit.from_qasm_str(circuit_data)
            coupling_map = CouplingMap(couplinglist=coupling_edges)

            pm = helix_pass_manager(coupling_map, params=params)
            t0 = time.perf_counter()
            compiled = pm.run(circuit)
            t1 = time.perf_counter()

            swaps = count_swaps(compiled)
            total_swaps += swaps
            total_depth += compiled.depth()
            total_time += (t1 - t0) * 1000
            per_case[name] = swaps
        except Exception:
            per_case[name] = 999999
            total_swaps += 999999

    return SweepResult(
        params=params,
        total_swaps=total_swaps,
        total_depth=total_depth,
        total_time_ms=total_time,
        per_case=per_case,
    )


def _serialize_cases(cases: list[BenchmarkCase]) -> list[tuple]:
    """Serialize benchmark cases for multiprocessing transfer."""
    serialized = []
    for case in cases:
        circuit = case.circuit_fn(*case.circuit_args)
        from qiskit.qasm2 import dumps
        qasm = dumps(circuit)
        edges = case.coupling_map.get_edges()
        serialized.append((case.name, qasm, edges, case.coupling_map.size()))
    return serialized


def run_sweep(
    cases: Optional[list[BenchmarkCase]] = None,
    config: Optional[SweepConfig] = None,
    n_workers: Optional[int] = None,
    verbose: bool = True,
) -> list[SweepResult]:
    """Run a parameter sweep across the benchmark suite.

    Parameters
    ----------
    cases : list[BenchmarkCase], optional
        Benchmark cases. Uses default suite if None.
    config : SweepConfig, optional
        Sweep grid configuration. Uses defaults if None.
    n_workers : int, optional
        Number of parallel workers. Defaults to cpu_count().
    verbose : bool
        Print progress updates.

    Returns
    -------
    list[SweepResult]
        Results sorted by total_swaps (best first).
    """
    if cases is None:
        cases = default_benchmark_suite()
    if config is None:
        config = SweepConfig()
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    param_grid = config.grid()
    if verbose:
        print(f"Sweep: {len(param_grid)} configurations x {len(cases)} cases "
              f"= {len(param_grid) * len(cases)} evaluations")
        print(f"Using {n_workers} workers")

    # Serialize cases for multiprocessing
    cases_data = _serialize_cases(cases)
    work_items = [(p, cases_data) for p in param_grid]

    t0 = time.perf_counter()
    if n_workers <= 1:
        results = [_evaluate_params(item) for item in work_items]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_evaluate_params, work_items)
    t1 = time.perf_counter()

    results.sort(key=lambda r: r.total_swaps)

    if verbose:
        print(f"\nSweep completed in {t1-t0:.1f}s")
        print(f"\nTop 5 configurations:")
        for i, r in enumerate(results[:5]):
            p = r.params
            print(f"  #{i+1}: {r.total_swaps} SWAPs — "
                  f"omega={p.omega}, c_log={p.c_log}, "
                  f"twist={p.twist_fraction}, k={p.k}, alpha={p.alpha}")

    return results


def refine_sweep(
    best_params: HelixParams,
    cases: Optional[list[BenchmarkCase]] = None,
    n_workers: Optional[int] = None,
    verbose: bool = True,
) -> list[SweepResult]:
    """Fine-grained sweep around a known good configuration.

    Takes the best params from a coarse sweep and searches a narrow
    neighborhood with finer granularity.
    """
    config = SweepConfig(
        omega_range=[best_params.omega * f for f in [0.7, 0.85, 1.0, 1.15, 1.3]],
        c_log_range=[best_params.c_log * f for f in [0.7, 0.85, 1.0, 1.15, 1.3]],
        twist_fraction_range=[
            max(0.1, best_params.twist_fraction - 0.05),
            best_params.twist_fraction,
            min(0.9, best_params.twist_fraction + 0.05),
        ],
        k_range=[max(2, best_params.k - 2), best_params.k, best_params.k + 2],
        alpha_range=[
            max(0.5, best_params.alpha - 1.0),
            best_params.alpha,
            best_params.alpha + 1.0,
        ],
    )
    return run_sweep(cases=cases, config=config, n_workers=n_workers, verbose=verbose)
