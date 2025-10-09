import numpy as np

from consts import TARGET_POSITION, STOCHASTIC_SPAWN_POSITIONS

def _cartesian_distance(a: list[float], b: list[float]) -> float:
    if np.array_equal(a, b):
        return 0.0
    return np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def fitness(spawn_point: list[float], history: list[tuple[float, float, float]]) -> float:

    end_to_target = _cartesian_distance(history[-1], TARGET_POSITION)

    start_to_spawn = _cartesian_distance(STOCHASTIC_SPAWN_POSITIONS[0], spawn_point)

    return -(end_to_target + start_to_spawn)