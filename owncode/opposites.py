import networkx as nx

_OPPOSITE = {
    "LEFT": "RIGHT", "RIGHT": "LEFT",
    "FRONT": "BACK", "BACK": "FRONT",
    "TOP": "BOTTOM", "BOTTOM": "TOP",
}

def _find_core(G: nx.DiGraph) -> int:
    # Prefer node with type='CORE', else 0, else the first node
    for n, d in G.nodes(data=True):
        if str(d.get("type", "")).upper() == "CORE":
            return n
    if 0 in G.nodes: 
        return 0
    return next(iter(G.nodes))

def faces_directly_from_core(G: nx.DiGraph) -> list[str]:
    """Return the list of face labels on edges *directly out of* the core."""
    c = _find_core(G)
    faces = []
    for child in G.successors(c):           # only edges leaving the core
        face = str(G[c][child].get("face", "")).upper()
        if face:
            faces.append(face)
    return faces

def has_core_opposite_pair(G: nx.DiGraph) -> bool:
    """True if core has at least one pair of opposite faces among its outgoing edges."""
    faces = set(faces_directly_from_core(G))
    return any(_OPPOSITE.get(f) in faces for f in faces)

def print_core_faces(G: nx.DiGraph) -> None:
    c = _find_core(G)
    print(f"Core node: {c} | faces out: {faces_directly_from_core(G)}")
