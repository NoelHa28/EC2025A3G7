from __future__ import annotations
import json, os
import numpy as np
from typing import Iterable

ENABLE_DYNAMIC: bool = True
SHORT: int = 15
MID: int = 45
LONG: int = 100
FIXED_DURATION_WHEN_DISABLED: int = 40

FINISH_X: float = 5.0
RUGGED_START_X: float = 0.4
UPHILL_START_X: float = 2.2

CURRENT_DURATION: int = SHORT


def set_enabled(flag: bool) -> None:
    """Turn dynamic duration on/off globally."""
    global ENABLE_DYNAMIC
    ENABLE_DYNAMIC = bool(flag)


def get_current_duration() -> int:
    """Read by evaluate(); returns fixed duration if dynamic is disabled."""
    return CURRENT_DURATION if ENABLE_DYNAMIC else FIXED_DURATION_WHEN_DISABLED


def _gate_distances_from_spawn_x(spawn_x: float) -> tuple[float, float, float]:
    """Distances (in meters) from this spawn’s x to the gates (rugged/uphill/finish)."""
    to_rugged = max(0.0, RUGGED_START_X - spawn_x)
    to_uphill = max(0.0, UPHILL_START_X - spawn_x)
    to_finish = max(0.0, FINISH_X - spawn_x)
    return to_rugged, to_uphill, to_finish


def _d0(spawn_xyz: list[float], target_xyz: list[float]) -> float:
    """Euclidean distance from spawn to target."""
    return float(
        np.linalg.norm(np.array(target_xyz, float) - np.array(spawn_xyz, float))
    )


def _covered_from_fitnesses(fitnesses: Iterable[float], d0_val: float) -> list[float]:
    """
    Map fitness -> covered distance using: covered = max(0, d0 + f),
    given fitness = -distance_to_target.
    """
    return [max(0.0, d0_val + float(f)) for f in fitnesses]


def _candidate_from_counts(
    covered: list[float], to_rugged: float, to_uphill: float, n_required: int
) -> int:
    """Choose SHORT/MID/LONG based on how many individuals reached the gates."""
    c_rugged = sum(c >= to_rugged for c in covered)
    c_uphill = sum(c >= to_uphill for c in covered)
    if c_uphill >= n_required:
        return LONG
    if c_rugged >= n_required:
        return MID
    return SHORT


def _candidate_from_best(
    covered_best: float, to_rugged: float, to_uphill: float
) -> int:
    """Choose SHORT/MID/LONG based on a single best covered value."""
    if covered_best >= to_uphill:
        return LONG
    if covered_best >= to_rugged:
        return MID
    return SHORT


def _publish_duration(candidate: int, *, monotonic: bool) -> int:
    """
    Write CURRENT_DURATION (with optional no-shrink), mirror to env, and return it.
    """
    global CURRENT_DURATION
    if monotonic:
        candidate = max(CURRENT_DURATION, candidate)
    CURRENT_DURATION = candidate
    os.environ["A3_CURRENT_DURATION"] = json.dumps(CURRENT_DURATION)
    return CURRENT_DURATION


def update_by_counts(
    prev_fitnesses: list[float] | None,
    spawn_xyz: list[float],
    *,
    n_required: int = 3,
    target_xyz: list[float] = [5.0, 0.0, 0.5],
    monotonic: bool = True,
) -> int:
    """
    Duration increases only when at least n_required individuals (from the PREVIOUS gen)
    have demonstrated reaching each gate.
    """
    if not ENABLE_DYNAMIC:
        return _publish_duration(FIXED_DURATION_WHEN_DISABLED, monotonic=False)

    if not prev_fitnesses:  # first gen or no data
        return _publish_duration(SHORT, monotonic=monotonic)

    to_rugged, to_uphill, _ = _gate_distances_from_spawn_x(spawn_xyz[0])
    d0_val = _d0(spawn_xyz, target_xyz)
    covered = _covered_from_fitnesses(prev_fitnesses, d0_val)
    candidate = _candidate_from_counts(covered, to_rugged, to_uphill, n_required)
    return _publish_duration(candidate, monotonic=monotonic)


def update_by_best(
    best_fitness_reference: float | None,
    spawn_xyz: list[float],
    *,
    target_xyz: list[float] = [5.0, 0.0, 0.5],
    monotonic: bool = True,
) -> int:
    """
    Duration based on a single reference fitness:
      - pass the global best-so-far, or
      - pass the best-of-last-generation if you prefer that behavior.
    """
    if not ENABLE_DYNAMIC:
        return _publish_duration(FIXED_DURATION_WHEN_DISABLED, monotonic=False)

    if best_fitness_reference is None:
        return _publish_duration(SHORT, monotonic=monotonic)

    to_rugged, to_uphill, _ = _gate_distances_from_spawn_x(spawn_xyz[0])
    d0_val = _d0(spawn_xyz, target_xyz)
    covered_best = max(0.0, d0_val + float(best_fitness_reference))
    candidate = _candidate_from_best(covered_best, to_rugged, to_uphill)
    return _publish_duration(candidate, monotonic=monotonic)
