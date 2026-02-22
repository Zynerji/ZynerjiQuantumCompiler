# ZynerjiQuantumCompiler

## Overview
Spectral graph matching for quantum circuit compilation using Dual-Helix Laplacians.
Targets 10%+ SWAP reduction over Qiskit `opt_level=3` on heavy-hex topologies.

## Architecture
```
DAGCircuit → interaction_graph(dag) → 4 Laplacians (2 circuit x 2 hardware)
  → k eigenvectors each → spectral_coords → cost_matrix → Hungarian LAP
  → initial_layout → SabreSwap routing → optimized circuit
```

## Key Modules
- `zynerji_qc/core/dual_helix.py` — Sparse Dual-Helix engine (adapted from HelicalATPG)
- `zynerji_qc/core/interaction_graph.py` — DAGCircuit → weighted adjacency
- `zynerji_qc/core/hardware_graph.py` — CouplingMap → adjacency + error weighting
- `zynerji_qc/core/spectral_match.py` — Cost matrix + Hungarian LAP solver
- `zynerji_qc/passes/helix_layout.py` — Qiskit AnalysisPass (sets layout)
- `zynerji_qc/passes/helix_routing.py` — Qiskit TransformationPass (SabreSwap with helix layout)
- `zynerji_qc/pipeline.py` — helix_pass_manager() factory
- `zynerji_qc/bench.py` — Benchmark harness (circuit zoo x topology zoo)
- `zynerji_qc/sweep.py` — Parameter sweep + auto-tuner

## Constants
- PHI = (1 + sqrt(5)) / 2 (golden ratio)
- Default params: omega=0.3, c_log=1.0, twist_fraction=0.33, k=8, alpha=3.0

## Running
```bash
pip install -e ".[dev]"
pytest tests/ -v
python scripts/compare_qiskit.py
python scripts/run_sweep.py --topology heavy_hex_127
```
