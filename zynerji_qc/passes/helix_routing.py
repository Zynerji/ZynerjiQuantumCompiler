"""Qiskit TransformationPass that routes using SabreSwap with the Helix layout."""

from __future__ import annotations

from qiskit.transpiler import TransformationPass, CouplingMap
from qiskit.transpiler.passes import SabreSwap
from qiskit.dagcircuit import DAGCircuit


class HelixRouting(TransformationPass):
    """Route a circuit using SabreSwap seeded with the Helix-computed layout.

    Expects ``property_set["layout"]`` to be set by a prior HelixLayout pass.
    Applies SabreSwap with the "decay" heuristic for SWAP insertion.
    """

    def __init__(self, coupling_map: CouplingMap, heuristic: str = "decay"):
        """
        Parameters
        ----------
        coupling_map : CouplingMap
            Hardware qubit connectivity.
        heuristic : str
            SabreSwap heuristic: "basic", "lookahead", or "decay".
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.heuristic = heuristic

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run SabreSwap routing with the Helix-computed initial layout."""
        sabre = SabreSwap(self.coupling_map, heuristic=self.heuristic)
        sabre.property_set = self.property_set
        return sabre.run(dag)
