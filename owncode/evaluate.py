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

def evaluate(robot: Robot) -> float:
    """
    Evaluate a robot genotype by simulating it and returning fitness.
    Handles invalid or unstable robots safely.
    """
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

        # Run simulation
        experiment(
            robot=robot,
            core=core,
            controller=ctrl,
            mode="simple",
            duration=10
        )

        # --- SAFETY CHECKS ---
        # No trajectory recorded
        if not tracker.history["xpos"] or len(tracker.history["xpos"][0]) < 5:
            return -100

        traj = np.array(tracker.history["xpos"][0])

        # NaN or Inf in trajectory
        if np.any(~np.isfinite(traj)):
            return -100

        # Explosion: robot goes absurdly far away
        if np.max(np.abs(traj)) > 1000:
            return -100

        # Otherwise compute normal fitness
        f = fitness(traj)
        steps_recorded = len(tracker.history["xpos"][0]) if tracker.history["xpos"] else 0
        console.log(f"Sim ran for {steps_recorded} steps")

        return float(f)

    except Exception as e:
        console.log(f"Simulation failed: {e}")
        mj.set_mjcb_control(None)  # always clean up
        return -100