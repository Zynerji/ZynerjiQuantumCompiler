"""Tests for spectral graph matching."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from qiskit.transpiler import CouplingMap, Layout
from qiskit.circuit import QuantumRegister

from zynerji_qc.core.dual_helix import HelixParams, compute_spectral_coords
from zynerji_qc.core.spectral_match import build_cost_matrix, spectral_match


def _make_ring_adj(n: int) -> csr_matrix:
    rows, cols, vals = [], [], []
    for i in range(n):
        j = (i + 1) % n
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([1.0, 1.0])
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def _make_path_adj(n: int) -> csr_matrix:
    rows, cols, vals = [], [], []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
        vals.extend([1.0, 1.0])
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


class TestBuildCostMatrix:
    def test_same_graph_low_diagonal(self):
        """Cost of matching a graph to itself should have low diagonal."""
        adj = _make_ring_adj(8)
        params = HelixParams(k=4)
        coords = compute_spectral_coords(adj, params)
        cost = build_cost_matrix(coords, coords)
        # Diagonal should be zero (same coordinates)
        assert np.allclose(np.diag(cost), 0.0, atol=1e-10)

    def test_rectangular_cost_matrix(self):
        """Cost matrix for n_logical < n_physical should be rectangular."""
        small = _make_ring_adj(4)
        large = _make_ring_adj(8)
        params = HelixParams(k=3)
        c_small = compute_spectral_coords(small, params)
        c_large = compute_spectral_coords(large, params)
        cost = build_cost_matrix(c_small, c_large)
        assert cost.shape == (4, 8)

    def test_cost_non_negative(self):
        adj = _make_ring_adj(6)
        params = HelixParams(k=3)
        coords = compute_spectral_coords(adj, params)
        cost = build_cost_matrix(coords, coords)
        assert np.all(cost >= -1e-10)

    def test_empty_graphs(self):
        adj = csr_matrix((0, 0))
        params = HelixParams()
        coords = compute_spectral_coords(adj, params)
        cost = build_cost_matrix(coords, coords)
        assert cost.shape == (0, 0)


class TestSpectralMatch:
    def test_basic_matching(self):
        """Match a 4-qubit circuit to a 4-qubit ring."""
        circuit_adj = _make_path_adj(4)
        cm = CouplingMap.from_ring(4)
        hw_adj = _make_ring_adj(4)
        qreg = QuantumRegister(4, "q")

        layout = spectral_match(circuit_adj, hw_adj, qreg, cm)

        assert isinstance(layout, Layout)
        # All 4 logical qubits should be mapped
        physical_qubits = set()
        for i in range(4):
            phys = layout[qreg[i]]
            assert 0 <= phys < 4
            physical_qubits.add(phys)
        # All mapped to distinct physical qubits
        assert len(physical_qubits) == 4

    def test_rectangular_matching(self):
        """Match 4 logical qubits to 8 physical qubits."""
        circuit_adj = _make_path_adj(4)
        cm = CouplingMap.from_ring(8)
        hw_adj = _make_ring_adj(8)
        qreg = QuantumRegister(4, "q")

        layout = spectral_match(circuit_adj, hw_adj, qreg, cm)

        physical_qubits = set()
        for i in range(4):
            phys = layout[qreg[i]]
            assert 0 <= phys < 8
            physical_qubits.add(phys)
        assert len(physical_qubits) == 4

    def test_empty_circuit(self):
        circuit_adj = csr_matrix((0, 0))
        cm = CouplingMap.from_ring(4)
        hw_adj = _make_ring_adj(4)
        qreg = QuantumRegister(0, "q")

        layout = spectral_match(circuit_adj, hw_adj, qreg, cm)
        assert isinstance(layout, Layout)

    def test_custom_params(self):
        circuit_adj = _make_path_adj(4)
        cm = CouplingMap.from_ring(6)
        hw_adj = _make_ring_adj(6)
        qreg = QuantumRegister(4, "q")

        params = HelixParams(omega=0.5, k=4, alpha=5.0)
        layout = spectral_match(circuit_adj, hw_adj, qreg, cm, params)

        physical_qubits = set()
        for i in range(4):
            phys = layout[qreg[i]]
            physical_qubits.add(phys)
        assert len(physical_qubits) == 4
