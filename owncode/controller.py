import numpy as np
import mujoco as mj
import numpy.typing as npt

from robot import Robot

def cpg(
    model: mj.MjModel,
    data: mj.MjData,
    robot: Robot,
) -> npt.NDArray[np.float64]:
    return robot.brain.step().astype(np.float64)

def random(
    model: mj.MjModel,
    data: mj.MjData,
    robot: Robot,
) -> npt.NDArray[np.float64]:
    return np.random.uniform(-np.pi / 2, np.pi / 2, model.nu)