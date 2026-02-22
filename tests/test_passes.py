"""Tests for Qiskit transpiler passes (HelixLayout, HelixRouting)."""

import pytest

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, Layout, PassManager

from zynerji_qc.core.dual_helix import HelixParams
from zynerji_qc.passes.helix_layout import HelixLayout
from zynerji_qc.passes.helix_routing import HelixRouting


def _make_bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _make_ghz_circuit(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def _make_multi_cx_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(0, 3)
    return qc


class TestHelixLayout:
    def test_sets_layout(self):
        """HelixLayout should set property_set['layout']."""
        cm = CouplingMap.from_ring(4)
        qc = _make_multi_cx_circuit()
        dag = circuit_to_dag(qc)

        pass_ = HelixLayout(cm)
        pass_.run(dag)

        assert "layout" in pass_.property_set
        layout = pass_.property_set["layout"]
        assert isinstance(layout, Layout)

    def test_layout_maps_all_qubits(self):
        cm = CouplingMap.from_ring(6)
        qc = _make_ghz_circuit(4)
        dag = circuit_to_dag(qc)

        pass_ = HelixLayout(cm)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        qreg = list(dag.qregs.values())[0]
        physical_qubits = set()
        for i in range(4):
            phys = layout[qreg[i]]
            assert 0 <= phys < 6
            physical_qubits.add(phys)
        assert len(physical_qubits) == 4

    def test_trivial_circuit(self):
        """Single-qubit circuit should get trivial layout."""
        cm = CouplingMap.from_ring(4)
        qc = QuantumCircuit(1)
        qc.h(0)
        dag = circuit_to_dag(qc)

        pass_ = HelixLayout(cm)
        pass_.run(dag)
        assert "layout" in pass_.property_set

    def test_custom_params(self):
        cm = CouplingMap.from_ring(6)
        qc = _make_ghz_circuit(4)
        dag = circuit_to_dag(qc)

        params = HelixParams(omega=0.5, k=4, alpha=5.0)
        pass_ = HelixLayout(cm, params=params)
        pass_.run(dag)
        assert "layout" in pass_.property_set

    def test_depth_weighted(self):
        cm = CouplingMap.from_ring(6)
        qc = _make_ghz_circuit(4)
        dag = circuit_to_dag(qc)

        pass_ = HelixLayout(cm, depth_weighted=True)
        pass_.run(dag)
        assert "layout" in pass_.property_set


class TestHelixRouting:
    def test_routing_produces_valid_circuit(self):
        """HelixRouting should produce a valid routed circuit."""
        cm = CouplingMap.from_ring(4)
        qc = _make_multi_cx_circuit()
        dag = circuit_to_dag(qc)

        # First run layout
        layout_pass = HelixLayout(cm)
        layout_pass.run(dag)

        # Then run routing (share property_set)
        routing_pass = HelixRouting(cm)
        routing_pass.property_set = layout_pass.property_set
        routed_dag = routing_pass.run(dag)

        # Should return a valid DAG
        assert routed_dag is not None
        routed_circuit = dag_to_circuit(routed_dag)
        assert routed_circuit.num_qubits >= 4
