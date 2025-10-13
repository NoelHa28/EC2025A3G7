from typing import Literal
from pathlib import Path
from collections.abc import Callable

import mujoco as mj
import numpy as np

from ariel import console
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph

from ea import BodyEA, MindEA

import controller
from robot import Robot
from simulate import experiment
from visualise import plot_evolution_progress

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

def simulate_best_robot(best_robot: Robot, mode: ViewerTypes = "launcher", controller_func: Callable = controller.cpg) -> None:
    """Simulate the best evolved robot."""
    console.log("Simulating best evolved robot...")
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Construct robot
    core = construct_mjspec_from_graph(best_robot.graph)

    # Setup tracker
    tracker = Tracker(
        mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
    )

    # Setup controller
    ctrl = Controller(
        controller_callback_function=controller_func,
        tracker=tracker,
    )

    # Run simulation with visualization
    experiment(robot=best_robot, core=core, controller=ctrl, mode=mode, duration=120)

    np.save('', tracker.history['xpos'][0])

def main() -> None:
    """Entry point - Run evolutionary algorithm to evolve robots."""
    console.log("Starting robot evolution...")

    # Body evolution parameters
    body_params = {
        "population_size": 7,
        "generations": 400,
        "mutation_rate": 0.4,
        "crossover_rate": 0.9,
        "crossover_type": "uniform",
        "elitism": 1,
        "selection": "tournament",
        "tournament_size": 4,
        "dynamic_duration": True,
        "dynamic_duration_n_required": 3,
        "top_k_re_eval": 3,

    }

    # Mind evolution parameters
    mind_params = {
        "population_size": 50,
        "generations": 10,
        "crossover_rate": 0.5,
        "crossover_type": "uniform",
        "elitism": 5,
        "selection": "tournament",
        "tournament_size": 5,
    }

    ea = BodyEA(
        body_params=body_params,
        mind_params=mind_params,
    )

    console.log(f"Population size: {body_params['population_size']}")
    console.log(f"Generations: {body_params['generations']}")
    console.log(f"Selection: {ea.selection}")
    console.log(f"Crossover: {ea.crossover.crossover_type}")

    # Run evolution
    console.log("Evolution started...")
    best_genes, best_fitness_history, avg_fitness_history = ea.run()
    best_robot = Robot(body_genotype=best_genes)

    # Show evolution progress
    console.log("Evolution completed!")
    console.log(f"Best fitness achieved: {max(best_fitness_history):.4f}")

    # Plot evolution progress
    plot_evolution_progress(best_fitness_history, avg_fitness_history)

    # Simulate and visualize the best robot
    simulate_best_robot(best_robot, mode="launcher")

def run_mind():
    ea = MindEA(
        robot=Robot.load_robot(),
        population_size=50,
        generations=20,
        crossover_rate=0.5,
        crossover_type="uniform",
        elitism=5,
        selection="tournament",
        tournament_size=5,
    )
    best_genes, best_fitness_history, avg_fitness_history = ea.run()
    print(f"Best fitness achieved: {max(best_fitness_history):.4f}")

def run_best():
    best_robot = Robot.load_robot()
    simulate_best_robot(best_robot, mode='simple')

if __name__ == "__main__":
    run_best()
    # main()