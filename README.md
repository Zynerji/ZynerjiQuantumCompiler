# ZynerjiQuantumCompiler

Spectral graph matching for quantum circuit compilation. Uses dual-helix Laplacians to embed qubit interaction graphs into a shared spectral space with hardware coupling maps, producing initial layouts that reduce SWAP gate insertions during routing.

## Results

**6.5% fewer SWAPs than Qiskit `optimization_level=3`** across a 12-case benchmark suite (8-32 qubits). Never loses to Qiskit in any individual case.

| Circuit | Topology | Helix | Qiskit3 | Diff | Improvement |
|---------|----------|------:|--------:|-----:|------------:|
| QFT-32 | 8x8 grid | 280 | 311 | -31 | 10.0% |
| Random-32 | 8x8 grid | 238 | 250 | -12 | 4.8% |
| QFT-16 | 6x6 grid | 61 | 70 | -9 | 12.9% |
| QV-16 | heavy-hex 19q | 146 | 152 | -6 | 3.9% |
| QFT-8 | 4x4 grid | 9 | 11 | -2 | 18.2% |
| QFT-8 | heavy-hex 19q | 17 | 19 | -2 | 10.5% |
| QAOA-16 | heavy-hex 19q | 15 | 17 | -2 | 11.8% |
| Random-16 | 6x6 grid | 64 | 66 | -2 | 3.0% |
| Random-8 | 4x4 grid | 12 | 14 | -2 | 14.3% |
| QFT-16 | heavy-hex 19q | 106 | 106 | 0 | 0.0% |
| QAOA-8 | heavy-hex 19q | 4 | 4 | 0 | 0.0% |
| QV-8 | heavy-hex 19q | 19 | 19 | 0 | 0.0% |
| **Total** | | **971** | **1039** | **-68** | **6.5%** |

### Large-Scale (QFT-64)

Qiskit results are best-of-5 seeds for fair comparison.

| Circuit | Topology | Helix | Qiskit3 | Diff | Improvement |
|---------|----------|------:|--------:|-----:|------------:|
| QFT-64 | 12x12 grid (144q) | 817 | 853 | -36 | 4.2% |
| QFT-64 | heavy-hex d=7 (115q) | 1458 | 1505 | -47 | 3.1% |

All results are deterministic (`seed=42`) and reproducible.

## How It Works

```
DAGCircuit --> interaction_graph(dag) --> dual-helix Laplacians
    --> k eigenvectors --> spectral_coords --> cost_matrix --> Hungarian LAP
    --> initial_layout --> SabreSwap routing --> optimized circuit
```

The compiler generates a diverse ensemble of layout candidates from multiple strategies:

1. **Greedy distance-aware placement** - Places qubits one at a time, minimizing weighted hardware distance to already-placed neighbors
2. **Spectral matching** - Dual-helix and standard Laplacian embeddings with parameter diversity
3. **Hall spectral placement** - Fiedler vector as x-coordinate, 3rd eigenvector as y-coordinate (VLSI placement theory)
4. **Sequential band placement** - Maps Fiedler-ordered qubits onto hardware's longest geodesic chain
5. **Angular (phase) matching** - Cosine similarity cost matrix for magnitude-invariant structural matching
6. **Signed interaction graphs** - Ternary {-1, 0, +1} adjacency with repulsive edges for competing qubits
7. **Golden-ratio eigenvector selection** - Multi-scale spectral information without harmonic redundancy
8. **Q-factor amplification** - Antinode detection amplifies structurally ideal qubit-position matches
9. **Hierarchical multi-scale matching** - Recursive Fiedler bisection with coarse-to-fine refinement
10. **SABRE layout** - Qiskit's stochastic iterative layout as an ensemble member
11. **Qiskit baseline** - Multi-seed Qiskit `opt_level` 2 and 3 (guarantees never worse)

After deduplication, all candidates are routed and the best is selected.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+, Qiskit >= 1.0, NumPy, SciPy.

## Usage

### As a drop-in replacement for Qiskit transpile

```python
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from zynerji_qc.pipeline import helix_transpile

# Your circuit
qc = QuantumCircuit(16)
# ... add gates ...

# Hardware topology
coupling_map = CouplingMap.from_grid(6, 6)

# Compile
compiled = helix_transpile(qc, coupling_map)
```

### As a Qiskit PassManager

```python
from zynerji_qc.pipeline import helix_pass_manager

pm = helix_pass_manager(coupling_map)
compiled = pm.run(qc)
```

### With custom parameters

```python
from zynerji_qc.core.dual_helix import HelixParams

params = HelixParams(
    omega=0.3,           # Phase frequency
    c_log=1.0,           # Log spacing constant
    twist_fraction=0.33, # Mobius twist threshold
    k=8,                 # Eigenvectors per helix
    alpha=3.0,           # Spectral attenuation
    use_helix=True,      # Dual-helix vs standard Laplacian
)

compiled = helix_transpile(qc, coupling_map, params=params)
```

## Benchmarks

```bash
# Standard benchmark suite (12 cases, ~5 minutes)
python scripts/compare_qiskit.py

# Large QFT benchmarks (QFT-64, QFT-128)
python scripts/bench_large_qft.py

# Run all tests
pytest tests/ -v
```

## Project Structure

```
zynerji_qc/
    core/
        dual_helix.py         # Sparse Dual-Helix engine (cos/sin Laplacians)
        interaction_graph.py  # Circuit --> weighted adjacency matrix
        hardware_graph.py     # CouplingMap --> adjacency matrix
        spectral_match.py     # Cost matrices + Hungarian LAP matching
    passes/
        helix_layout.py       # Qiskit AnalysisPass
        helix_routing.py      # Qiskit TransformationPass
    pipeline.py               # Multi-strategy ensemble transpiler
    bench.py                  # Benchmark harness
    sweep.py                  # Parameter sweep + auto-tuner
tests/                        # 50 unit tests
scripts/
    compare_qiskit.py         # Head-to-head vs Qiskit opt_level=3
    bench_large_qft.py        # QFT-64/128 benchmarks
paper/
    whitepaper.tex            # Technical whitepaper (LaTeX)
    whitepaper.pdf            # Compiled PDF
```

## Trade-offs

The ensemble approach is computationally expensive. QFT-32 compiles in ~90 seconds vs Qiskit's ~0.3 seconds. This is a deliberate trade-off: compilation time for circuit quality. For circuits where SWAP reduction directly improves execution fidelity on noisy hardware (variational algorithms, error correction, hardware characterization), the trade-off is worthwhile.

## Theory

The dual-helix Laplacian constructs two phase-modulated graph Laplacians using cos and sin basis functions with golden-ratio coupling. The phase modulation creates "stereoscopic vision" --- two independent spectral views of graph structure. The Mobius twist adds topological sensitivity for distant node pairs.

Spectral attenuation (`alpha=3.0`) emphasizes low-frequency eigenvectors that capture global graph structure over high-frequency modes that represent local noise.

See [paper/whitepaper.pdf](paper/whitepaper.pdf) for full technical details.

## License

MIT
