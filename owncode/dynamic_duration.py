from __future__ import annotations
import json, os
import numpy as np

ENABLE_DYNAMIC: bool = True
SHORT: int = 15  # up to rugged start
MID: int = 45  # up to uphill start (rugged completed)
LONG: int = 100  # to the finish
FIXED_DURATION_WHEN_DISABLED: int = 40  # used when ENABLE_DYNAMIC=False

FINISH_X: float = 5.0
RUGGED_START_X: float = 0.7
UPHILL_START_X: float = 2.5

CURRENT_DURATION: int = SHORT


def _estimate_distance_covered_from_fitness(
    best_fitness_so_far: float, spawn_xyz: list[float], target_xyz: list[float]
) -> float:
    """
    Your fitness = -distance(target, current).
    Let d0 = distance(target, spawn).
    If f = -d, then distance_covered ≈ d0 - d = d0 - (-f) = d0 + f.
    """
    target = np.array(target_xyz, dtype=float)
    spawn = np.array(spawn_xyz, dtype=float)
    d0 = float(np.linalg.norm(target - spawn))  # ~5.8 from (-0.8,0,0.1) to (5,0,0.5)
    return max(0.0, d0 + float(best_fitness_so_far))


def _checkpoint_distances(spawn_x: float) -> tuple[float, float, float]:
    """
    Distances along x from current spawn to {rugged start, uphill start, finish}.
    We treat these as 'progress gates' to grow the allowed duration.
    """
    to_rugged = max(0.0, RUGGED_START_X - spawn_x)  # ~1.5 from -0.8 → 0.7
    to_uphill = max(0.0, UPHILL_START_X - spawn_x)  # ~3.3 from -0.8 → 2.5
    to_finish = max(0.0, FINISH_X - spawn_x)  # ~5.8 from -0.8 → 5.0
    return to_rugged, to_uphill, to_finish


def compute_duration_for_generation(
    best_fitness_so_far: float | None,
    spawn_xyz: list[float],
    target_xyz: list[float] = [5.0, 0.0, 0.5],
) -> int:
    """
    Decide the duration this generation should use.
    - If dynamic is OFF → fixed duration.
    - If no best yet → SHORT.
    - Else compare estimated progress to the three gates.
    """
    global ENABLE_DYNAMIC, SHORT, MID, LONG, FIXED_DURATION_WHEN_DISABLED

    if not ENABLE_DYNAMIC:
        return FIXED_DURATION_WHEN_DISABLED

    if best_fitness_so_far is None:
        return SHORT

    # Progress in meters (rough estimate from your fitness)
    covered = _estimate_distance_covered_from_fitness(
        best_fitness_so_far, spawn_xyz, target_xyz
    )

    # Gates from current spawn
    to_rugged, to_uphill, to_finish = _checkpoint_distances(spawn_xyz[0])

    if covered < to_rugged:
        return SHORT
    elif covered < to_uphill:
        return MID
    else:
        # once we've shown we can reach the uphill, give the full budget to finish
        return LONG


def set_enabled(flag: bool) -> None:
    """Public toggle."""
    global ENABLE_DYNAMIC
    ENABLE_DYNAMIC = bool(flag)


def update_for_generation(
    best_fitness_so_far: float|None, spawn_xyz: list[float]
) -> int:
    """
    Compute & store CURRENT_DURATION (read by evaluate()).
    Returns the new duration for logging.
    """
    global CURRENT_DURATION
    CURRENT_DURATION = compute_duration_for_generation(best_fitness_so_far, spawn_xyz)
    # Optional: also export to env so new subprocesses see it immediately
    os.environ["A3_CURRENT_DURATION"] = json.dumps(CURRENT_DURATION)
    return CURRENT_DURATION


def get_current_duration() -> int:
    """Used inside evaluate() to avoid hard-coding a duration."""
    global CURRENT_DURATION, ENABLE_DYNAMIC, FIXED_DURATION_WHEN_DISABLED
    return CURRENT_DURATION if ENABLE_DYNAMIC else FIXED_DURATION_WHEN_DISABLED
