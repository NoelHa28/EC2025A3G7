from typing import Any, Literal
from pathlib import Path
import mujoco as mj
from mujoco import viewer
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

# Local libraries
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena

from robot import Robot

# Type Aliases
ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables

SPAWN_POS = [-0.8, 0, 0.1]


def experiment(
    robot: Robot,
    core: CoreModule,
    controller: Controller,
    duration: int = 40,
    mode: ViewerTypes = "viewer",
    spawn_pos: list[float] | None = None,
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
    world.spawn(core.spec, spawn_position=(spawn_pos or SPAWN_POS).copy())

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
    args: list[Any] = [robot]  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
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
