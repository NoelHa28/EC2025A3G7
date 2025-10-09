from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np

from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer
from robot import Robot

type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

SEED = 42
RNG = np.random.default_rng(SEED)

DATA = Path.cwd() / "__data__" / __file__.split("/")[-1][:-3]
DATA.mkdir(exist_ok=True)

SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]

def show_xpos_history(robot: Robot, history: list[float], save_path_=None) -> None:
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

    x0 = robot.spawn_point[0]
    y0 = robot.spawn_point[1]

    # Plot x,y trajectory
    # ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(x0, y0, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    if save_path_ is not None:
        plt.savefig(save_path_)
    else:
        # Show results
        plt.show()


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
