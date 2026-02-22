"""Tests for interaction graph extraction from quantum circuits."""

import numpy as np
import pytest

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from zynerji_qc.core.interaction_graph import build_interaction_graph


class TestBuildInteractionGraph:
    def test_empty_circuit(self):
        qc = QuantumCircuit(0)
        adj = build_interaction_graph(qc)
        assert adj.shape == (0, 0)

    def test_single_cx(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        adj = build_interaction_graph(qc)
        assert adj.shape == (2, 2)
        assert adj[0, 1] == 1.0
        assert adj[1, 0] == 1.0
        assert adj[0, 0] == 0.0

    def test_multiple_cx(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        adj = build_interaction_graph(qc)
        assert adj[0, 1] == 2.0  # Two CX between 0-1
        assert adj[1, 2] == 1.0  # One CX between 1-2
        assert adj[0, 2] == 0.0  # No CX between 0-2

    def test_symmetric(self):
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 3)
        qc.cx(1, 2)
        adj = build_interaction_graph(qc)
        dense = adj.toarray()
        assert np.allclose(dense, dense.T)

    def test_only_1q_gates(self):
        """Single-qubit gates should not create edges."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)
        qc.rz(0.5, 2)
        adj = build_interaction_graph(qc)
        assert adj.nnz == 0

    def test_dag_input(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        adj = build_interaction_graph(dag)
        assert adj[0, 1] == 1.0

    def test_depth_weighted(self):
        """Earlier gates should have higher weights with depth weighting."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)  # Layer 0: weight 1/(1+0) = 1.0
        qc.cx(1, 2)  # Layer 1: weight 1/(1+1) = 0.5
        adj = build_interaction_graph(qc, depth_weighted=True)
        assert adj[0, 1] > adj[1, 2]

    def test_depth_weighted_same_layer(self):
        """Gates in the same layer should have equal weights."""
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 3)
        adj = build_interaction_graph(qc, depth_weighted=True)
        assert abs(adj[0, 1] - adj[2, 3]) < 1e-10

    def test_multi_qubit_circuit(self):
        """A circuit with many CX gates should produce a dense interaction graph."""
        qc = QuantumCircuit(4)
        # GHZ-like pattern
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(0, 2)
        adj = build_interaction_graph(qc)
        assert adj.shape == (4, 4)
        assert adj.nnz > 0
        # 0-1, 1-2, 2-3, 0-2 should all have edges
        assert adj[0, 1] >= 1.0
        assert adj[1, 2] >= 1.0
        assert adj[2, 3] >= 1.0
        assert adj[0, 2] >= 1.0
