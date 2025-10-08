import numpy as np
from pathlib import Path
from opposites import _find_core, faces_directly_from_core, print_core_faces, has_core_opposite_pair

from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
    draw_graph
)

from nde import NeuralDevelopmentalEncodingWithLoading
from cpg import EvolvableCPG

DATA = Path.cwd() / "__data__"

NUM_OF_MODULES = 30

class Robot:
    def __init__(
        self,
        body_genotype: list[np.ndarray] | None = None,
        graph: None | object = None,
        mind_genotype: np.ndarray | None = None
    ) -> None:
        
        assert body_genotype is not None or graph is not None, "Either body_genotype or graph must be provided."
        self.body_genotype = body_genotype

        if body_genotype is not None:

            NDE = NeuralDevelopmentalEncodingWithLoading(number_of_modules=NUM_OF_MODULES)

            if Path('nde_model.pt').exists():
                NDE.load('nde_model.pt')

            NDE.eval()
            self.graph = HighProbabilityDecoder(NUM_OF_MODULES).probability_matrices_to_graph(*NDE.forward(self.body_genotype))

        elif graph is not None:
            self.graph = graph

        self.mind_genotype = mind_genotype

        self._number_of_hinges = sum(1 for n in self.graph.nodes if self.graph.nodes[n]["type"] == "HINGE")
        self.brain = EvolvableCPG(self._number_of_hinges)

        if self.mind_genotype is not None:
            self.brain.set_genotype(self.mind_genotype)
        
    
    def save(self) -> None:
        save_graph_as_json(self.graph, DATA / "best_robot_graph.json")
        np.save(DATA / "best_robot_body.npy", self.body_genotype)
        np.save(DATA / "best_robot_brain.npy", self.mind_genotype)

    @classmethod
    def load_robot(
        cls,
        path_to_graph: Path = DATA / "best_robot_body.npy",
        path_to_brain: Path = DATA / "best_robot_brain.npy"
    ) -> None:
        return cls(body_genotype=np.load(path_to_graph), mind_genotype=np.load(path_to_brain))
