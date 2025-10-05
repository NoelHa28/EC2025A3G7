import numpy as np
from pathlib import Path

from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)

from nde import NeuralDevelopmentalEncodingWithLoading
from cpg import EvolvableCPG

DATA = Path.cwd() / "__data__"

NUM_OF_MODULES = 30

class Robot:
    def __init__(
        self,
        body_genotype: list[np.ndarray],
        mind_genotype: np.ndarray | None = None
    ) -> None:
        
        self.body_genotype = body_genotype
        self.mind_genotype = mind_genotype

        NDE = NeuralDevelopmentalEncodingWithLoading(number_of_modules=NUM_OF_MODULES)

        if Path('nde_model.pt').exists():
            NDE.load('nde_model.pt')

        NDE.eval()
        HPD = HighProbabilityDecoder(NUM_OF_MODULES)

        self.graph = HPD.probability_matrices_to_graph(*NDE.forward(self.body_genotype))
        self._number_of_hinges = sum(1 for n in self.graph.nodes if self.graph.nodes[n]["type"] == "HINGE")
        if self._number_of_hinges == 0:
            raise RuntimeError("No hinges in the robot, cannot build brain.")
        
        self.brain = EvolvableCPG(self._number_of_hinges)

        if self.mind_genotype is not None:
            self.brain.set_genotype(self.mind_genotype)
    
    def save(self) -> None:
        save_graph_as_json(self.graph, DATA / "best_robot_graph.json")
        
        # if not self._brain: # TODO: Uncomment this when brain saving is implemented
        #     raise Exception("No brain to save!")

        # save_brain_weights = self._brain.get_weights() # TODO: Finish this saving, also with the architecture sizes
    