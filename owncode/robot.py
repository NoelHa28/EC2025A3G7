from typing import Any
import numpy as np
from pathlib import Path

from ariel.simulation.controllers.hopfs_cpg import HopfCPG
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
    draw_graph,
)

from brain import NNBrain
from networkx import DiGraph
from nde import NeuralDevelopmentalEncodingWithLoading
from cpg import CPG

DATA = Path.cwd() / "__data__"

NUM_OF_MODULES = 30

# NDE = NeuralDevelopmentalEncodingWithLoading(number_of_modules=NUM_OF_MODULES)

# if Path('nde_model.pt').exists():
#     NDE.load('nde_model.pt')

# NDE.eval()
# HPD = HighProbabilityDecoder(NUM_OF_MODULES)

class Robot:
    def __init__(self, genes: list[np.ndarray], weights: np.ndarray | None = None) -> None:
        
        self.genes = genes

        NDE = NeuralDevelopmentalEncodingWithLoading(number_of_modules=NUM_OF_MODULES)

        if Path('nde_model.pt').exists():
            NDE.load('nde_model.pt')

        NDE.eval()
        HPD = HighProbabilityDecoder(NUM_OF_MODULES)

        self.graph = HPD.probability_matrices_to_graph(*NDE.forward(self.genes))
        self._number_of_hinges = sum(1 for n in self.graph.nodes if self.graph.nodes[n]["type"] == "HINGE")
        self.brain = self.build_brain(self._number_of_hinges * 3 + 6 * 3 + 1)

        if weights is not None and isinstance(self.brain, NNBrain):
            self.brain.set_weights(weights)

    def build_cpg(self) -> CPG:
        return CPG(self._number_of_hinges)

    def build_brain(self, input_size: int, hidden_size: int = 16) -> NNBrain:
        return NNBrain(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=self._number_of_hinges,
        )

    def save(self) -> None:
        save_graph_as_json(self.graph, DATA / "best_robot_graph.json")
        
        # if not self._brain: # TODO: Uncomment this when brain saving is implemented
        #     raise Exception("No brain to save!")

        # save_brain_weights = self._brain.get_weights() # TODO: Finish this saving, also with the architecture sizes
    