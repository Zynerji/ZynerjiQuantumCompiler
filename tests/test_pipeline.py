"""Tests for the full Helix compilation pipeline."""

import pytest

from qiskit import QuantumCircuit
from qiskit.synthesis.qft import synth_qft_full
from qiskit.transpiler import CouplingMap

from zynerji_qc.core.dual_helix import HelixParams
from zynerji_qc.pipeline import helix_pass_manager


class TestHelixPassManager:
    def test_basic_transpilation(self):
        """End-to-end: transpile a circuit through the Helix pipeline."""
        cm = CouplingMap.from_ring(6)
        pm = helix_pass_manager(cm)

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)

        result = pm.run(qc)
        assert result is not None
        assert result.num_qubits == 6  # Expanded to full device

    def test_qft_transpilation(self):
        """QFT should transpile successfully."""
        cm = CouplingMap.from_ring(8)
        pm = helix_pass_manager(cm)
        qc = synth_qft_full(4)
        result = pm.run(qc)
        assert result is not None
        assert result.num_qubits == 8

    def test_custom_params(self):
        cm = CouplingMap.from_ring(6)
        params = HelixParams(omega=0.5, k=4, alpha=2.0)
        pm = helix_pass_manager(cm, params=params)

        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 3)
        result = pm.run(qc)
        assert result is not None

    def test_heavy_hex_topology(self):
        """Should work with heavy-hex topology."""
        cm = CouplingMap.from_heavy_hex(3)
        pm = helix_pass_manager(cm)

        qc = QuantumCircuit(4)
        qc.h(0)
        for i in range(3):
            qc.cx(i, i + 1)
        result = pm.run(qc)
        assert result is not None

    def test_grid_topology(self):
        cm = CouplingMap.from_grid(3, 3)
        pm = helix_pass_manager(cm)

        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        result = pm.run(qc)
        assert result is not None
        assert result.num_qubits == 9

    def test_no_1q_optimization(self):
        cm = CouplingMap.from_ring(4)
        pm = helix_pass_manager(cm, optimize_1q=False)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = pm.run(qc)
        assert result is not None

    def test_single_qubit_circuit(self):
        """Single-qubit circuit should pass through without error."""
        cm = CouplingMap.from_ring(4)
        pm = helix_pass_manager(cm)

        qc = QuantumCircuit(1)
        qc.h(0)
        result = pm.run(qc)
        assert result is not None
