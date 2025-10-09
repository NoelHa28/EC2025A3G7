import sys
from typing import Literal
from pathlib import Path
import random

import mujoco as mj
import numpy as np

from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import draw_graph
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker

from ea import BodyEA, MindEA
from robot import Robot
from fitness_function import fitness
from visualise import plot_evolution_progress, show_xpos_history
from morphology_constraints import is_robot_viable
from simulate import experiment
from evaluate import evaluate
import controller
import random
from consts import RNG, GENOTYPE_SIZE


# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

def simulate_best_robot(best_robot: Robot, mode: ViewerTypes = "launcher") -> None:
    """Simulate the best evolved robot."""
    console.log("Simulating best evolved robot...")
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Construct robot
    core = construct_mjspec_from_graph(best_robot.graph)

    best_robot.save()

    # Setup tracker
    tracker = Tracker(
        mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
    )

    # Setup controller
    ctrl = Controller(
        controller_callback_function=controller.cpg,
        tracker=tracker,
    )

    # Run simulation with visualization
    experiment(robot=best_robot, core=core, controller=ctrl, mode=mode, duration=120)

    # Show trajectory
    if tracker.history["xpos"] and len(tracker.history["xpos"][0]) > 0:
        show_xpos_history(tracker.history["xpos"][0])
        f = fitness(tracker.history["xpos"][0])
        console.log(f"Best robot fitness: {f:.4f}")
    else:
        console.log("No movement data recorded!")


def main() -> None:
    """Entry point - Run evolutionary algorithm to evolve robots."""
    console.log("Starting robot evolution...")

    # Body evolution parameters
    body_params = {
        "population_size": 5,  # Smaller population for faster testing
        "generations": 20,  # More generations for better evolution
        "genotype_size": 64,
        "mutation_rate": 0.2,  # Lower mutation rate to preserve good solutions
        "crossover_rate": 0.8,  # Higher crossover rate
        "crossover_type": "uniform",  # Good for real-valued genes
        "elitism": 1,  # Keep 1 best individual (10% of population)
        "selection": "tournament",  # Tournament selection
        "tournament_size": 2,
        "dynamic_duration": True,  # Enable dynamic duration
        "dynamic_duration_n_required": 3,  # Require 3 individuals to reach gates
    }

    # Mind evolution parameters
    mind_params = {
        "population_size": 5,
        "generations": 10,
        "mutation_rate": 0.5,
        "crossover_rate": 0.0,
        "crossover_type": "onepoint",
        "elitism": 1,
        "selection": "tournament",
        "tournament_size": 2,
    }

    # Create evolutionary algorithm
    ea = BodyEA(
        body_params=body_params,
        mind_params=mind_params,
        evaluator=evaluate,
    )

    console.log(f"Population size: {body_params['population_size']}")
    console.log(f"Generations: {body_params['generations']}")
    console.log(f"Selection: {ea.selection}")
    console.log(f"Crossover: {ea.crossover.crossover_type}")

    # Run evolution
    console.log("Evolution started...")
    best_genes, best_fitness_history, avg_fitness_history = ea.run()
    best_robot = Robot(best_genes)

    # Show evolution progress
    console.log("Evolution completed!")
    console.log(f"Best fitness achieved: {max(best_fitness_history):.4f}")

    # Plot evolution progress
    plot_evolution_progress(best_fitness_history, avg_fitness_history)

    # Simulate and visualize the best robot
    simulate_best_robot(best_robot, mode="launcher")

def main_single_robot() -> None:
    robot = None
    while robot is None or not is_robot_viable(robot, max_bricks_per_limb=3):
        genotype = [
            RNG.uniform(0, 1, GENOTYPE_SIZE).astype(np.float32),
            RNG.uniform(-100, 100, GENOTYPE_SIZE).astype(np.float32),
            RNG.uniform(-100, 100, GENOTYPE_SIZE).astype(np.float32),
        ]
        robot = Robot(
            spawn_point=[-0.8, 0, 0.1],
            body_genotype=genotype
            )

    f = evaluate(robot)
    print(f"Single robot fitness: {f:.4f}")

def run_mindEA() -> None:

    robot = None
    while robot is None or not is_robot_viable(robot, max_bricks_per_limb=3):
        genotype = [
            RNG.uniform(0, 1, GENOTYPE_SIZE).astype(np.float32),
            RNG.uniform(-100, 100, GENOTYPE_SIZE).astype(np.float32),
            RNG.uniform(-100, 100, GENOTYPE_SIZE).astype(np.float32),
        ]
        robot = Robot(genotype)

    ea = MindEA(
        robot=robot,
        population_size=200,
        generations=15,
        mutation_rate=0.9,
        crossover_rate=0.0,
        crossover_type="blend",
        elitism=20,
        selection="tournament",
        tournament_size=3,
    )

    # Run evolution
    console.log("Evolution started...")
    best_genes, best_fitness_history, avg_fitness_history = ea.run()
    robot.brain.set_genotype(best_genes)

    # Show evolution progress
    console.log("Evolution completed!")
    console.log(f"Best fitness achieved: {max(best_fitness_history):.4f}")

    # Plot evolution progress
    plot_evolution_progress(best_fitness_history, avg_fitness_history)

    # Calculate the slope from begin to end
    if len(avg_fitness_history) > 1:
        slope = (avg_fitness_history[-1] - avg_fitness_history[0]) / (
            len(avg_fitness_history) - 1
        )
        console.log(f"Slope of avg fitness: {slope:.4f}")
    else:
        console.log("Not enough data to calculate slope.")

    # Simulate and visualize the best robot
    simulate_best_robot(robot)


if __name__ == "__main__":
    main()
    # main_single_robot()
    # run_mindEA()
