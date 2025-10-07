import json
import os
import sys
import numpy as np
from robot import Robot
import mujoco as mj

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker

# Import evolutionary algorithm
from robot import Robot
from fitness_function import fitness
from simulate import experiment
import controller

SEED = 42
RNG = np.random.default_rng(SEED)

# STOCHASTIC_SPAWN_POSITIONS = [
#     [-0.8, 0, 0.1], # starting position
#     [0.7, 0, 0.1],  # rugged terrain
#     [2.5, 0, 0.1],  # uphill
# ]

STOCHASTIC_SPAWN_POSITIONS = [
    [-0.8, 0, 0.1], # starting position
]
CURRENT_SPAWN: list[float] | None = None

def set_spawn_position(spawn: list[float]) -> None:
    """Set the global spawn position for evaluation."""
    global CURRENT_SPAWN
    CURRENT_SPAWN = spawn
    os.environ["A3_SPAWN"] = json.dumps(spawn) # For subprocesses


def evaluate(robot: Robot, spawn=None) -> float:
    """
    Evaluate a robot genotype by simulating it and returning fitness.
    Handles invalid or unstable robots safely.
    """
    if spawn is not None:
        chosen_spawn = spawn
    elif CURRENT_SPAWN is not None:
        chosen_spawn = CURRENT_SPAWN
    else:
        # try env (for multiprocessing workers)
        env_spawn = os.environ.get("A3_SPAWN")
        if env_spawn:
            try:
                chosen_spawn = json.loads(env_spawn)
            except Exception:
                chosen_spawn = STOCHASTIC_SPAWN_POSITIONS[RNG.integers(0, len(STOCHASTIC_SPAWN_POSITIONS))]
        else:
            chosen_spawn = STOCHASTIC_SPAWN_POSITIONS[RNG.integers(0, len(STOCHASTIC_SPAWN_POSITIONS))]

    try:
        mj.set_mjcb_control(None)  # reset MuJoCo state

        core = construct_mjspec_from_graph(robot.graph)

        # Tracker to monitor root
        tracker = Tracker(
            mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM,
            name_to_bind="core",
        )

        # Simple controller for stability (can swap with nn_controller later)
        ctrl = Controller(
            controller_callback_function=controller.cpg,
            tracker=tracker,
        )

        print(f"[EVAL] pid={os.getpid()} spawn_used={chosen_spawn}", flush=True)

        # Run simulation
        experiment(
            robot=robot,
            core=core,
            controller=ctrl,
            mode="simple",
            duration=10,
            spawn_pos=chosen_spawn,
        )

        # --- SAFETY CHECKS ---
        # No trajectory recorded
        if not tracker.history["xpos"] or len(tracker.history["xpos"][0]) < 5:
            return -100

        traj = np.array(tracker.history["xpos"][0])

        # NaN or Inf in trajectory
        if np.any(~np.isfinite(traj)):
            return -100

        # Robot shouldn't learn to fall
        if all(np.isclose(start_cord, end_cord, atol=0.02) for start_cord, end_cord in zip(traj[5], traj[-1])):
            return -100

        # Otherwise compute normal fitness
        f = fitness(traj)
        steps_recorded = (
            len(tracker.history["xpos"][0]) if tracker.history["xpos"] else 0
        )
        console.log(f"Sim ran for {steps_recorded} steps")

        return float(f)

    except Exception as e:
        console.log(f"Simulation failed: {e}")
        mj.set_mjcb_control(None)  # always clean up
        return -100
