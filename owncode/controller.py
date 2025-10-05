import numpy as np
import numpy.typing as npt
import mujoco as mj

from robot import Robot


def simple(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
    """
    Simple sine controller that drives each actuator with a sinusoidal torque.
    """
    n = model.nu  # number of actuators
    t = data.time

    # Tunable parameters
    frequency = np.random.uniform(0.5, 1.5)  # Hz
    amplitude = np.pi / 2  # full-range amplitude
    phase_offset = np.linspace(0, np.pi, n)  # staggered phases per actuator

    ctrl = amplitude * np.sin(2 * np.pi * frequency * t + phase_offset)
    return ctrl.astype(np.float64)

def cpg(
    model: mj.MjModel,
    data: mj.MjData,
    robot: Robot,
) -> npt.NDArray[np.float64]:

    return robot.brain.step().astype(np.float64)  # type: ignore
