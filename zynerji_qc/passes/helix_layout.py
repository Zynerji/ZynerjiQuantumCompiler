"""Qiskit AnalysisPass that computes an initial layout via Dual-Helix spectral matching."""

from __future__ import annotations

from typing import Optional

from qiskit.transpiler import AnalysisPass, CouplingMap, Target
from qiskit.dagcircuit import DAGCircuit

from zynerji_qc.core.dual_helix import HelixParams
from zynerji_qc.core.interaction_graph import build_interaction_graph
from zynerji_qc.core.hardware_graph import build_hardware_graph
from zynerji_qc.core.spectral_match import spectral_match


class HelixLayout(AnalysisPass):
    """Compute initial qubit layout using Dual-Helix spectral graph matching.

    This pass analyzes the circuit's two-qubit gate interaction pattern and
    the hardware coupling map's topology, computes spectral coordinates for
    both graphs using dual helical Laplacians, and finds an optimal matching
    via the Hungarian algorithm.

    Sets ``property_set["layout"]`` for downstream passes.
    """

    def __init__(
        self,
        coupling_map: CouplingMap,
        target: Optional[Target] = None,
        params: Optional[HelixParams] = None,
        depth_weighted: bool = False,
        error_weighted: bool = False,
    ):
        """
        Parameters
        ----------
        coupling_map : CouplingMap
            Hardware qubit connectivity.
        target : Target, optional
            Backend target with error information.
        params : HelixParams, optional
            Dual-Helix tuning parameters.
        depth_weighted : bool
            Weight circuit interaction edges by gate depth.
        error_weighted : bool
            Weight hardware edges by gate fidelity.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.target = target
        self.params = params or HelixParams()
        self.depth_weighted = depth_weighted
        self.error_weighted = error_weighted

    def run(self, dag: DAGCircuit):
        """Run the Helix layout analysis pass."""
        n_qubits = dag.num_qubits()

        # For trivial circuits, skip spectral analysis
        if n_qubits <= 1:
            from qiskit.transpiler import Layout
            qregs = list(dag.qregs.values())
            if qregs:
                layout = Layout({qregs[0][i]: i for i in range(n_qubits)})
            else:
                layout = Layout()
            self.property_set["layout"] = layout
            return

        # Build interaction graph from circuit
        circuit_adj = build_interaction_graph(dag, depth_weighted=self.depth_weighted)

        # Build hardware graph from coupling map
        hardware_adj = build_hardware_graph(
            self.coupling_map,
            target=self.target,
            error_weighted=self.error_weighted,
        )

        # Get quantum register for Layout construction
        qregs = list(dag.qregs.values())
        if not qregs:
            return
        qreg = qregs[0]

        # Spectral matching
        layout = spectral_match(
            circuit_adj,
            hardware_adj,
            qreg,
            self.coupling_map,
            self.params,
        )

        self.property_set["layout"] = layout
