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

NUM_OF_MODULES = 25

import networkx as nx

def morphology_graph_difference(G1: nx.DiGraph, G2: nx.DiGraph) -> float:
    """
    Compute a normalized difference between two directed morphology graphs (0 = identical, 1 = completely different).
    Considers both node attributes and edge structure.
    """
    # --- 1. Node-level comparison ---
    all_nodes = set(G1.nodes) | set(G2.nodes)
    node_diffs = 0.0
    
    for node in all_nodes:
        a = G1.nodes.get(node)
        b = G2.nodes.get(node)

        # Missing node in one graph
        if a is None or b is None:
            node_diffs += 1.0
            continue

        # Type difference
        type_diff = 0.0 if a.get("type") == b.get("type") else 1.0

        # Rotation difference (0–180 normalized)
        rot_a = int(a.get("rotation", "DEG_0").split("_")[1])
        rot_b = int(b.get("rotation", "DEG_0").split("_")[1])
        rot_diff = abs(rot_a - rot_b) % 360
        rot_diff = min(rot_diff, 360 - rot_diff) / 180.0

        node_diffs += 0.7 * type_diff + 0.3 * rot_diff

    node_diff_norm = node_diffs / max(1, len(all_nodes))

    # --- 2. Edge-level comparison ---
    all_edges = set(G1.edges) | set(G2.edges)
    edge_diffs = 0.0

    for edge in all_edges:
        e1 = edge in G1.edges
        e2 = edge in G2.edges
        if e1 and e2:
            edge_diffs += 0.0
        else:
            edge_diffs += 1.0

    edge_diff_norm = edge_diffs / max(1, len(all_edges))

    # --- 3. Combine node + edge components ---
    total_diff = 0.6 * node_diff_norm + 0.4 * edge_diff_norm
    return float(min(1.0, total_diff))


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

            self.NDE = NeuralDevelopmentalEncodingWithLoading(number_of_modules=NUM_OF_MODULES)
            # self.NDE.save()
            if Path('nde_model.pt').exists():
                self.NDE.load('nde_model.pt')

            self.NDE.eval()
            self.graph = HighProbabilityDecoder(NUM_OF_MODULES).probability_matrices_to_graph(*self.NDE.forward(self.body_genotype))

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
