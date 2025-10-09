from itertools import combinations
import sys
import networkx as nx

from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import draw_graph
from robot import Robot

from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
import mujoco as mj
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker
import controller
from simulate import experiment


MIN_HINGES = 2
MIN_BRICKS = 2

ADJACENT_FACES_SET = {
    frozenset({"LEFT", "BACK", "BOTTOM"}),
    frozenset({"LEFT", "BACK", "TOP"}),
    frozenset({"LEFT", "FRONT", "BOTTOM"}),
    frozenset({"LEFT", "FRONT", "TOP"}),
    frozenset({"RIGHT", "BACK", "BOTTOM"}),
    frozenset({"RIGHT", "BACK", "TOP"}),
    frozenset({"RIGHT", "FRONT", "BOTTOM"}),
    frozenset({"RIGHT", "FRONT", "TOP"}),
}

_OPPOSITE_FACES = {
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
    "FRONT": "BACK",
    "BACK": "FRONT",
    "TOP": "BOTTOM",
    "BOTTOM": "TOP"
}

def limb_nodes(G: nx.DiGraph, root: int) -> set[int]:
    return {root} | nx.descendants(G, root)

def node_type(G: nx.DiGraph, n: int) -> str:
    return G.nodes[n].get("type", "NONE")

def _find_core(G: nx.DiGraph) -> int | None:
    core_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "CORE"]
    if len(core_nodes) == 0:
        print("[KILL] No CORE node found.")
        return None
    if len(core_nodes) > 1:
        print(f"[KILL] Multiple CORE nodes found: {core_nodes}.")
        return None
    return core_nodes[0]

def hit_minimal_modules(G: nx.DiGraph) -> bool:

    hinge_count = sum(1 for n in G.nodes if G.nodes[n]["type"] == "HINGE")
    brick_count = sum(1 for n in G.nodes if G.nodes[n]["type"] == "BRICK")

    return brick_count > MIN_BRICKS and hinge_count > MIN_HINGES

def has_adjacent_faces(G: nx.DiGraph, core: int) -> bool:
    adjacent_faces = {ed.get("face") for _, v, ed in list(G.out_edges(core, data=True))}
    if len(adjacent_faces) < 3:
        return False
    
    if "TOP" in adjacent_faces:
        return True

    if "BOTTOM" in adjacent_faces:
        return True
    
    if len(adjacent_faces) == 3:
        if frozenset(adjacent_faces) in ADJACENT_FACES_SET:
            print(f"[KILL] Core has has adjacent faces: {adjacent_faces}, limiting mobility.")
            return True
        
    if len(adjacent_faces) > 3:
        for combination in combinations(adjacent_faces, 3):
            if frozenset(combination) in ADJACENT_FACES_SET:
                print(f"[KILL] Core has adjacent faces: {combination}, limiting mobility.")
                return True
    return False

def is_valid_limb(G: nx.DiGraph, root: int, max_bricks_per_limb: int) -> bool:
    nodes = limb_nodes(G, root)
    blocks_map = {n: node_type(G, n) for n in nodes if node_type(G, n) in {"BRICK", "HINGE"}}

    block_count = len(blocks_map)
    hinge_count = sum(1 for n in blocks_map if blocks_map[n] == "HINGE")
    brick_count = block_count - hinge_count

    print(f"    Limb rooted at edge {root} has {block_count} blocks ({brick_count} BRICKs, {hinge_count} HINGEs).")
    
    reasons = []

    if brick_count > max_bricks_per_limb:
        reasons.append(f"more than {max_bricks_per_limb} BRICKs")
    if block_count < 3:
        reasons.append("fewer than 3 building blocks")
    if brick_count == block_count:
        reasons.append("only BRICKs")
    if brick_count > 4:
        reasons.append("more than 4 BRICKs")
    if hinge_count < 2:
        reasons.append("fewer than 2 HINGEs")
    # Brick nodes cannot be deeper than direct children of the core

    if reasons:
        # print(f"  -> INVALID limb: {', '.join(reasons)}")
        return False
    else:
        # print("  -> VALID limb")
        return True

def two_valid_limbs(G: nx.DiGraph, out_edges: list[tuple[int, dict]], max_bricks_per_limb: int) -> int:
    edge_name = None
    for connection_index, data in out_edges:
        if edge_name is None:
            edge_name = data.get("face")
        elif data.get("face") != _OPPOSITE_FACES[edge_name]:
            return False

        if not is_valid_limb(G, connection_index, max_bricks_per_limb):
            return False

    return True

def three_valid_limbs(G: nx.DiGraph, out_edges: list[tuple[int, dict]], max_bricks_per_limb: int) -> int:
    for connection_index, data in out_edges:
        if not is_valid_limb(G, connection_index, max_bricks_per_limb):
            return False
    return True

def four_valid_limbs(G: nx.DiGraph, out_edges: list[tuple[int, dict]], max_bricks_per_limb: int) -> int:
    for connection_index, data in out_edges:
        if not is_valid_limb(G, connection_index, max_bricks_per_limb):
            return False
    return True

def is_robot_viable(robot: Robot, max_bricks_per_limb: int = 3) -> bool:
    """
    Viability rules:
      Core:
        - must have at least 2 limbs attached NOT from the TOP face
        - must be able to move (at least one HINGE reachable from the core)
      Limb (rooted at a non-TOP edge out of the core):
        1) cannot be only BRICKs
        2) must have at least 3 building blocks (BRICK or HINGE)
        3) must have at least 2 HINGEs
        4) (optional) must not exceed max_bricks_per_limb bricks

    Args:
        robot: Robot with a DiGraph (robot.graph) where nodes have "type" and edges have "face"
        max_bricks_per_limb: If set, any limb with more than this many BRICKs is invalid.

    Prints detailed reasons when returning False and a summary per limb.
    """
    G: nx.DiGraph = robot.graph

    # ---- Find core ----
    core = _find_core(G)
    if core is None:
        return False
    
    # ---- Minimal modules check ----
    if not hit_minimal_modules(G):
        return False
    
    # ---- Adjacent faces from core ----
    if has_adjacent_faces(G, core):
        return False

    # ---- Limb roots (non-TOP edges out of core) ----
    out_edges = [(connection_index, data) for _, connection_index, data in G.out_edges(core, data=True)]

    match len(out_edges):
        case 0:
            print("[KILL] Core has no limbs attached.")
            return False
        case 1:
            print("[KILL] Core has only 1 limb attached.")
            return False
        case 2:
            if not two_valid_limbs(G, out_edges, max_bricks_per_limb):
                return False
        case 3:
            if not three_valid_limbs(G, out_edges, max_bricks_per_limb):
                return False
        case 4:
            if not four_valid_limbs(G, out_edges, max_bricks_per_limb):
                return False
        case _:
            return False

    print(f"[OK] Robot is viable: mobility OK")
    return True