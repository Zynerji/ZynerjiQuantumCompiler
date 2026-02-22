"""Benchmark harness for comparing Helix layout vs Qiskit defaults.

Generates a zoo of circuits and hardware topologies, transpiles each
with both Helix and Qiskit opt_level=0..3, and reports SWAP counts,
circuit depth, and compilation time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QuantumVolume, EfficientSU2
from qiskit.synthesis.qft import synth_qft_full
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap, PassManager

from zynerji_qc.core.dual_helix import HelixParams
from zynerji_qc.pipeline import helix_pass_manager, helix_transpile


# --- Circuit Zoo ---

def make_qft(n_qubits: int) -> QuantumCircuit:
    """QFT circuit on n qubits (decomposed to 2Q gates)."""
    return synth_qft_full(n_qubits)


def make_quantum_volume(n_qubits: int, depth: int = 0) -> QuantumCircuit:
    """Quantum Volume circuit (decomposed to 2Q gates)."""
    d = depth if depth > 0 else n_qubits
    qv = QuantumVolume(n_qubits, depth=d, seed=42)
    return qv.decompose().decompose()


def make_random(n_qubits: int, depth: int = 20) -> QuantumCircuit:
    """Random circuit with 2Q gates."""
    return random_circuit(n_qubits, depth, max_operands=2, seed=42)


def make_qaoa_maxcut(n_qubits: int, p: int = 1) -> QuantumCircuit:
    """QAOA MaxCut ansatz on a random 3-regular graph."""
    qc = QuantumCircuit(n_qubits)
    rng = np.random.default_rng(42)

    # Build random 3-regular-ish graph edges
    edges = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if rng.random() < 3.0 / (n_qubits - 1):
                edges.append((i, j))

    for layer in range(p):
        gamma = 0.5 + 0.1 * layer
        beta = 0.3 + 0.1 * layer
        # Cost layer
        for i, j in edges:
            qc.cx(i, j)
            qc.rz(2 * gamma, j)
            qc.cx(i, j)
        # Mixer layer
        for i in range(n_qubits):
            qc.rx(2 * beta, i)

    return qc


def make_vqe_hwe(n_qubits: int, reps: int = 2) -> QuantumCircuit:
    """Hardware-efficient VQE ansatz."""
    return EfficientSU2(n_qubits, reps=reps, entanglement="full")


# --- Topology Zoo ---

def make_heavy_hex(n: int) -> CouplingMap:
    """Heavy-hex coupling map (IBM-style).

    Approximate: generates a hex lattice then adds heavy edges.
    For exact IBM maps, use CouplingMap.from_heavy_hex(distance).
    """
    return CouplingMap.from_heavy_hex(n)


def make_grid(rows: int, cols: int) -> CouplingMap:
    """Square grid coupling map."""
    return CouplingMap.from_grid(rows, cols)


def make_ring(n: int) -> CouplingMap:
    """Ring coupling map."""
    return CouplingMap.from_ring(n)


def make_linear(n: int) -> CouplingMap:
    """Linear chain coupling map."""
    return CouplingMap.from_line(n)


# --- Benchmark Definitions ---

@dataclass
class BenchmarkCase:
    """A single benchmark case: circuit + topology."""
    name: str
    circuit_fn: callable
    circuit_args: tuple
    topology_name: str
    coupling_map: CouplingMap


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    case_name: str
    method: str
    swap_count: int
    depth: int
    compile_time_ms: float
    n_qubits: int
    n_gates_original: int
    n_gates_compiled: int


def count_swaps(circuit: QuantumCircuit) -> int:
    """Count SWAP gates in a compiled circuit."""
    return sum(1 for inst in circuit.data if inst.operation.name == "swap")


def default_benchmark_suite() -> list[BenchmarkCase]:
    """Generate the default benchmark suite."""
    cases = []

    # QFT on heavy-hex
    for n in [8, 16, 32]:
        d = max(3, (n + 18) // 19)  # heavy-hex distance for enough qubits
        cm = make_heavy_hex(d)
        if cm.size() >= n:
            cases.append(BenchmarkCase(
                name=f"qft_{n}_heavyhex_{cm.size()}",
                circuit_fn=make_qft,
                circuit_args=(n,),
                topology_name=f"heavy_hex_{cm.size()}",
                coupling_map=cm,
            ))

    # QFT on grid
    for n in [8, 16, 32]:
        side = int(np.ceil(np.sqrt(n * 2)))
        cm = make_grid(side, side)
        if cm.size() >= n:
            cases.append(BenchmarkCase(
                name=f"qft_{n}_grid_{side}x{side}",
                circuit_fn=make_qft,
                circuit_args=(n,),
                topology_name=f"grid_{side}x{side}",
                coupling_map=cm,
            ))

    # Quantum Volume on heavy-hex
    for n in [8, 16, 32]:
        d = max(3, (n + 18) // 19)
        cm = make_heavy_hex(d)
        if cm.size() >= n:
            cases.append(BenchmarkCase(
                name=f"qv_{n}_heavyhex_{cm.size()}",
                circuit_fn=make_quantum_volume,
                circuit_args=(n,),
                topology_name=f"heavy_hex_{cm.size()}",
                coupling_map=cm,
            ))

    # Random circuits on grid
    for n in [8, 16, 32]:
        side = int(np.ceil(np.sqrt(n * 2)))
        cm = make_grid(side, side)
        if cm.size() >= n:
            cases.append(BenchmarkCase(
                name=f"random_{n}_grid_{side}x{side}",
                circuit_fn=make_random,
                circuit_args=(n, 20),
                topology_name=f"grid_{side}x{side}",
                coupling_map=cm,
            ))

    # QAOA on heavy-hex
    for n in [8, 16, 32]:
        d = max(3, (n + 18) // 19)
        cm = make_heavy_hex(d)
        if cm.size() >= n:
            cases.append(BenchmarkCase(
                name=f"qaoa_{n}_heavyhex_{cm.size()}",
                circuit_fn=make_qaoa_maxcut,
                circuit_args=(n,),
                topology_name=f"heavy_hex_{cm.size()}",
                coupling_map=cm,
            ))

    return cases


def run_benchmark(
    cases: Optional[list[BenchmarkCase]] = None,
    params: Optional[HelixParams] = None,
    qiskit_levels: tuple[int, ...] = (0, 1, 2, 3),
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """Run the full benchmark suite.

    For each case, transpiles with:
    - Helix pass manager (our method)
    - Qiskit opt_level=0,1,2,3 (baselines)

    Returns list of BenchmarkResult.
    """
    if cases is None:
        cases = default_benchmark_suite()
    if params is None:
        params = HelixParams()

    results = []

    for case in cases:
        circuit = case.circuit_fn(*case.circuit_args)
        n_gates_orig = len(circuit.data)
        n_qubits = circuit.num_qubits

        if verbose:
            print(f"\n--- {case.name} ({n_qubits}q, {n_gates_orig} gates) ---")

        # Helix (multi-trial)
        try:
            t0 = time.perf_counter()
            compiled = helix_transpile(
                circuit,
                coupling_map=case.coupling_map,
                params=params,
                trials=20,
            )
            t1 = time.perf_counter()
            swaps = count_swaps(compiled)
            results.append(BenchmarkResult(
                case_name=case.name,
                method="helix",
                swap_count=swaps,
                depth=compiled.depth(),
                compile_time_ms=(t1 - t0) * 1000,
                n_qubits=n_qubits,
                n_gates_original=n_gates_orig,
                n_gates_compiled=len(compiled.data),
            ))
            if verbose:
                print(f"  Helix:  {swaps} SWAPs, depth {compiled.depth()}, "
                      f"{(t1-t0)*1000:.1f}ms")
        except Exception as e:
            if verbose:
                print(f"  Helix:  FAILED ({e})")

        # Qiskit baselines
        for level in qiskit_levels:
            try:
                t0 = time.perf_counter()
                compiled = transpile(
                    circuit,
                    coupling_map=case.coupling_map,
                    optimization_level=level,
                    seed_transpiler=42,
                )
                t1 = time.perf_counter()
                swaps = count_swaps(compiled)
                results.append(BenchmarkResult(
                    case_name=case.name,
                    method=f"qiskit_opt{level}",
                    swap_count=swaps,
                    depth=compiled.depth(),
                    compile_time_ms=(t1 - t0) * 1000,
                    n_qubits=n_qubits,
                    n_gates_original=n_gates_orig,
                    n_gates_compiled=len(compiled.data),
                ))
                if verbose:
                    print(f"  Qiskit opt={level}: {swaps} SWAPs, depth {compiled.depth()}, "
                          f"{(t1-t0)*1000:.1f}ms")
            except Exception as e:
                if verbose:
                    print(f"  Qiskit opt={level}: FAILED ({e})")

    return results


def format_results_table(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a markdown table."""
    lines = [
        "| Case | Method | SWAPs | Depth | Time (ms) | Gates (orig→compiled) |",
        "|------|--------|-------|-------|-----------|----------------------|",
    ]
    for r in sorted(results, key=lambda x: (x.case_name, x.method)):
        lines.append(
            f"| {r.case_name} | {r.method} | {r.swap_count} | "
            f"{r.depth} | {r.compile_time_ms:.1f} | "
            f"{r.n_gates_original}→{r.n_gates_compiled} |"
        )
    return "\n".join(lines)
