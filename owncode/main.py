"""Assignment 3 template code."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# Import evolutionary algorithm
from evolutionary_algorithm import evolutionary_algorithm

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]


def fitness_function(history: list[float]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1] # type: ignore

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance


def evaluate_robot_genotype(genotype: list[np.ndarray]) -> float:
    """
    Evaluate a robot genotype by simulating it and returning fitness.
    
    Args:
        genotype: Robot genotype [type_p_genes, conn_p_genes, rot_p_genes]
        
    Returns:
        Fitness score (negative distance to target)
    """
    try:
        # CRITICAL: Reset MuJoCo control callback to ensure clean state
        mj.set_mjcb_control(None)
        
        # Convert genotype to robot
        nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
        p_matrices = nde.forward(genotype)

        # Decode the high-probability graph
        hpd = HighProbabilityDecoder(NUM_OF_MODULES)
        robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
            p_matrices[0],
            p_matrices[1], 
            p_matrices[2],
        )

        # Construct robot
        core = construct_mjspec_from_graph(robot_graph)

        # Setup tracker
        mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
        name_to_bind = "core"
        tracker = Tracker(
            mujoco_obj_to_find=mujoco_type_to_find,
            name_to_bind=name_to_bind,
        )

        # Setup controller
        ctrl = Controller(
            controller_callback_function=nn_controller,
            tracker=tracker,
        )

        # Run simulation 
        experiment(robot=core, controller=ctrl, mode="simple", duration=15)

        # Calculate fitness
        if tracker.history["xpos"] and len(tracker.history["xpos"][0]) > 0:
            fitness = fitness_function(tracker.history["xpos"][0])
        else:
            # If no position history, return failure fitness
            fitness = -100.0
        
        # CRITICAL: Clean up MuJoCo state after evaluation
        mj.set_mjcb_control(None)
        
        return fitness
        
    except Exception as e:
        # If anything goes wrong, return very bad fitness
        console.log(f"Simulation failed: {e}")
        # Ensure MuJoCo state is clean even on failure
        mj.set_mjcb_control(None)
        return -100.0


def show_xpos_history(history: list[float]) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        camera=camera,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    # Show results
    plt.show()


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    # Initialize the networks weights randomly
    # Normally, you would use the genes of an individual as the weights,
    # Here we set them randomly for simplicity.
    w1 = RNG.normal(loc=0.0138, scale=0.2, size=(input_size, hidden_size))
    w2 = RNG.normal(loc=0.0138, scale=0.2, size=(hidden_size, hidden_size))
    w3 = RNG.normal(loc=0.0138, scale=0.2, size=(hidden_size, output_size))

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))

    # Scale the outputs
    return outputs * np.pi/4


def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    # CRITICAL FIX: Pass a copy of SPAWN_POS to prevent in-place modification!
    world.spawn(robot.spec, spawn_position=SPAWN_POS.copy())

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "launcher":
            # This opens a liver viewer of the simulation
            viewer.launch(
                model=model,
                data=data,
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )
    # ==================================================================== #


def plot_evolution_progress(best_fitness_history: list[float], avg_fitness_history: list[float]) -> None:
    """Plot the evolution progress over generations."""
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history, 'b-', label='Best Fitness', linewidth=2)
    plt.plot(avg_fitness_history, 'r--', label='Average Fitness', linewidth=1)
    plt.xlabel('Generation')
    plt.ylabel('Fitness (negative distance)')
    plt.title('Evolution Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def simulate_best_robot(best_genotype: list[np.ndarray], mode: ViewerTypes = "launcher") -> None:
    """Simulate the best evolved robot."""
    console.log("Simulating best evolved robot...")
    mj.set_mjcb_control(None)  # DO NOT REMOVE
    # Convert genotype to robot
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(best_genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # Save the best robot graph
    save_graph_as_json(
        robot_graph,
        DATA / "best_robot_graph.json",
    )

    # Construct robot
    core = construct_mjspec_from_graph(robot_graph)

    # Setup tracker
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    # Setup controller
    ctrl = Controller(
        controller_callback_function=nn_controller,
        tracker=tracker,
    )

    # Run simulation with visualization
    experiment(robot=core, controller=ctrl, mode=mode, duration=15)

    # Show trajectory
    if tracker.history["xpos"] and len(tracker.history["xpos"][0]) > 0:
        show_xpos_history(tracker.history["xpos"][0])
        fitness = fitness_function(tracker.history["xpos"][0])
        console.log(f"Best robot fitness: {fitness:.4f}")
    else:
        console.log("No movement data recorded!")


def main() -> None:
    """Entry point - Run evolutionary algorithm to evolve robots."""
    console.log("Starting robot evolution...")
    
    # Evolutionary algorithm parameters
    genotype_size = 64
    population_size = 100  # Start small for testing
    generations = 5      
    
    # Create evolutionary algorithm
    ea = evolutionary_algorithm(
        population_size=population_size,
        generations=generations,
        genotype_size=genotype_size,
        evaluator=evaluate_robot_genotype,
        mutation_rate=0.2,
        crossover_rate=0.7,
        crossover_type="onepoint",  # Good for real-valued genes
        elitism=3,              # Keep 3 best individuals
        selection="roulette",  # Roulette or tournament
        tournament_size=3
    )
    
    console.log(f"Population size: {population_size}")
    console.log(f"Generations: {generations}")
    console.log(f"Selection: {ea.selection}")
    console.log(f"Crossover: {ea.crossover_type}")
    
    # Run evolution
    console.log("Evolution started...")
    best_robot, best_fitness_history, avg_fitness_history = ea.run()
    
    # Show evolution progress
    console.log("Evolution completed!")
    console.log(f"Best fitness achieved: {max(best_fitness_history):.4f}")
    
    # Plot evolution progress
    plot_evolution_progress(best_fitness_history, avg_fitness_history)
    
    # Simulate and visualize the best robot
    simulate_best_robot(best_robot, mode="video")


def main_single_robot() -> None:
    """Original main function - test a single random robot."""
    console.log("Testing single random robot...")
    
    # ? ------------------------------------------------------------------ #
    genotype_size = 64
    type_p_genes = RNG.random(genotype_size).astype(np.float32)
    conn_p_genes = RNG.random(genotype_size).astype(np.float32)
    rot_p_genes = RNG.random(genotype_size).astype(np.float32)

    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]

    # Simulate single robot
    simulate_best_robot(genotype, mode="video")


if __name__ == "__main__":
    main()
    #main_single_robot()
