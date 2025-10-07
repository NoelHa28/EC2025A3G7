import json, os
import numpy as np

SEED: int = 42
RNG = np.random.default_rng(SEED)

MULTISPAWN_ENABLED: bool = False  

STOCHASTIC_SPAWN_POSITIONS: list[list[float]] = [
    [-0.8, 0, 0.1],  # start / flat
    [0.7,  0, 0.1],  # rugged
    [2.5,  0, 0.1],  # uphill
]
DEFAULT_SPAWN: list[float] = [-0.8, 0, 0.1]  

CURRENT_SPAWN: list[float] | None = None

def set_spawn_position(spawn: list[float]) -> None:
    """Set the single spawn used for this generation."""
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


