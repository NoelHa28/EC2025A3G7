import networkx as nx

from robot import Robot

def limb_nodes(G: nx.DiGraph, root: int) -> set[int]:
    return {root} | nx.descendants(G, root)


def is_robot_viable(robot: Robot, max_bricks_per_limb: int | None = None) -> bool:
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

    def node_type(n: int) -> str:
        return G.nodes[n].get("type", "NONE")

    # ---- Find core ----
    core_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "CORE"]
    if len(core_nodes) == 0:
        print("[KILL] No CORE node found.")
        return False
    if len(core_nodes) > 1:
        print(f"[KILL] Multiple CORE nodes found: {core_nodes}.")
        return False
    core = core_nodes[0]
    
    # ------------------------
    # 1. Morphology check
    # ------------------------
    MIN_HINGES = 2
    MIN_BRICKS = 2

    hinge_count = sum(1 for n in robot.graph.nodes if robot.graph.nodes[n]["type"] == "HINGE")
    brick_count = sum(1 for n in robot.graph.nodes if robot.graph.nodes[n]["type"] == "BRICK")
    
    if hinge_count < MIN_HINGES:
        print("too few hinges")
        return -100  # too few hinges, skip simulation
    if brick_count < MIN_BRICKS:
        print("too few bricks")
        return -100  # too few bricks, skip simulation

    # ---- Mobility check: any HINGE reachable from core? ----
    descendants = nx.descendants(G, core)
    if not any(node_type(n) == "HINGE" for n in descendants):
        print("[KILL] Core cannot move: no reachable HINGE from the CORE.")
        return False

    # ---- Limb roots (non-TOP edges out of core) ----
    out_edges = list(G.out_edges(core, data=True))
    limb_roots = []
    for _, v, ed in out_edges:
        face = ed.get("face")
        if face != "TOP":
            limb_roots.append((v, face))

    if len(limb_roots) == 0:
        print("[KILL] Core has no limbs attached on non-TOP faces.")
        return False

    valid_limb_count = 0

    for root, face in limb_roots:
        nodes = limb_nodes(G, root)
        blocks = [n for n in nodes if node_type(n) in {"BRICK", "HINGE"}]
        block_count = len(blocks)
        hinge_count = sum(1 for n in blocks if node_type(n) == "HINGE")
        brick_count = block_count - hinge_count
        only_bricks = (block_count > 0) and (hinge_count == 0)

        # Summary
        extra = f", max_bricks={max_bricks_per_limb}" if max_bricks_per_limb is not None else ""
        print(f"[LIMB] Root {root} via {face}: blocks={block_count}, hinges={hinge_count}, bricks={brick_count}{extra}")

        reasons = []
        if only_bricks:
            reasons.append("contains only BRICKs (no HINGEs)")
        if block_count < 3:
            reasons.append("fewer than 3 building blocks")
        if hinge_count < 2:
            reasons.append("fewer than 2 HINGEs")
        if max_bricks_per_limb is not None and brick_count > max_bricks_per_limb:
            reasons.append(f"too many BRICKs ({brick_count} > {max_bricks_per_limb})")

        if reasons:
            print(f"  -> INVALID limb: {', '.join(reasons)}")
        else:
            print("  -> VALID limb")
            valid_limb_count += 1

    if valid_limb_count < 2:
        print(f"[KILL] Not enough valid limbs: {valid_limb_count} valid (need at least 2).")
        return False

    print(f"[OK] Robot is viable: mobility OK and {valid_limb_count} valid limbs (>= 2).")
    return True