import numpy as np
import dynamic_duration as dd
from robot import Robot
import mujoco as mj
from collections.abc import Callable

from ariel import console
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph

import controller
from robot import Robot
from simulate import experiment
from fitness_function import fitness

from consts import STOCHASTIC_SPAWN_POSITIONS


def evaluate(robot: Robot, learn_test: bool = False, controller_func: Callable[[], np.ndarray] = controller.cpg) -> float:
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
            controller_callback_function=controller_func,
            tracker=tracker,
        )

        if learn_test:
            # Run simulation
            experiment(
                robot=robot,
                core=core,
                controller=ctrl,
                mode="simple",
                duration=10,
                spawn_pos=STOCHASTIC_SPAWN_POSITIONS[0],  # fixed spawn for learning test
            )
            
            dist3 = tracker.history["xpos"][0][3]
            distend = tracker.history["xpos"][0][-1]
            dist_travelled = np.linalg.norm(np.array(distend) - np.array(dist3))
            console.log(f"Distance travelled in learning test: {dist_travelled:.4f}")
            if dist_travelled < 0.2:
                console.log("Robot failed learning test (did not move enough).")
                return False
            else:
                console.log("Robot passed learning test.")
                return True  # neutral fitness for passing learning test
        else:
            # Run simulation
            experiment(
                robot=robot,
                core=core,
                controller=ctrl,
                mode="simple",
                duration=dd.get_current_duration(),
                spawn_pos=robot.spawn_point,
            )

        # --- SAFETY CHECKS ---
        # No trajectory recorded
        if not tracker.history["xpos"] or len(tracker.history["xpos"][0]) < 5:
            return -100

        traj = np.array(tracker.history["xpos"][0])

        # NaN or Inf in trajectory
        if np.any(~np.isfinite(traj)):
            return -100

        # # Robot shouldn't learn to fall
        # if all(
        #     np.isclose(start_cord, end_cord, atol=0.02)
        #     for start_cord, end_cord in zip(traj[5], traj[-1])
        # ):
        #     return -100

        # Otherwise compute normal fitness
        f = fitness(robot.spawn_point, traj)
        steps_recorded = (
            len(tracker.history["xpos"][0]) if tracker.history["xpos"] else 0
        )
        # console.log(f"Sim ran for {steps_recorded} steps")

        return float(f)

    except Exception as e:
        console.log(f"Simulation failed: {e}")
        mj.set_mjcb_control(None)  # always clean up
        return -100
