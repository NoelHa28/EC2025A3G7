import json, os
import numpy as np

SEED: int = 42
RNG = np.random.default_rng(SEED)

MULTISPAWN_ENABLED: bool = False  

STOCHASTIC_SPAWN_POSITIONS: list[list[float]] = [
    [-0.8, 0, 0.1],  # start / flat
    [0.4,  0, 0.1],  # rugged
    [2.2,  0, 0.1],  # uphill
]
DEFAULT_SPAWN: list[float] = [-0.8, 0, 0.1]  
CURRENT_SPAWN: list[float] | None = None

def _load_spawn_from_env() -> list[float] | None:
    """Try to read a spawn set by the parent process (BodyEA) via env var."""
    s = os.environ.get("A3_SPAWN")
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

def current_or_env_default() -> list[float]:
    """
    Workers (spawned processes) won't see parent's CURRENT_SPAWN.
    Use env var if set, else default.
    """
    global CURRENT_SPAWN
    if CURRENT_SPAWN is not None:
        return CURRENT_SPAWN
    env_spawn = _load_spawn_from_env()
    if env_spawn is not None:
        CURRENT_SPAWN = env_spawn
        return CURRENT_SPAWN
    return DEFAULT_SPAWN

def set_spawn_position(spawn: list[float]) -> None:
    """Set the single spawn used for this generation. Also exports to env for workers"""
    global CURRENT_SPAWN
    CURRENT_SPAWN = list(spawn)  
    os.environ["A3_SPAWN"] = json.dumps(CURRENT_SPAWN)  # helps new worker processes

def set_current_spawn(multispawn: bool) -> None:
    """Choose and set the generation's spawn (one position)."""
    if multispawn:
        idx = RNG.integers(0, len(STOCHASTIC_SPAWN_POSITIONS))
        spawn = STOCHASTIC_SPAWN_POSITIONS[idx]
    else:
        spawn = DEFAULT_SPAWN
    set_spawn_position(spawn)
    return CURRENT_SPAWN


