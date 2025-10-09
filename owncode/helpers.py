import json
import networkx as nx
from pathlib import Path

import numpy as np

from consts import RNG, GENOTYPE_SIZE

def load_graph(source: str | Path | list[dict]) -> nx.DiGraph:
    """
    Load a directed graph from JSON data of the format:
    [
        {
            "directed": true,
            "nodes": [...],
            "edges": [...]
        }
    ]
    Returns a networkx.DiGraph with node and edge attributes preserved.
    """
    # Parse if file path or JSON string
    if isinstance(source, (str, Path)):
        if Path(source).exists():
            with open(source, "r") as f:
                data = json.load(f)
        else:
            data = json.loads(source)
    else:
        data = source  # already parsed JSON
    
    # The top-level list may contain multiple graphs; take first
    g_data = data[0] if isinstance(data, list) else data
    
    directed = g_data.get("directed", True)
    G = nx.DiGraph() if directed else nx.Graph()

    # Add nodes
    for node in g_data.get("nodes", []):
        node_id = node.pop("id")
        G.add_node(node_id, **node)

    # Add edges
    for edge in g_data.get("edges", []):
        src = edge.pop("source")
        tgt = edge.pop("target")
        G.add_edge(src, tgt, **edge)

    return G