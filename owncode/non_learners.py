import mujoco as mj
import numpy as np

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments.olympic_arena import OlympicArena
from ariel.utils.tracker import Tracker
from robot import Robot
from simulate import experiment


class ViabilityConfig:
    duration_s: float = 1.5
    treshold_distance: float = 0.05 
    probe_seed: int = 42
    probe_amplitude: float = np.pi/3
    probe_switch_s: float = 0.2
    probe_settle_s: float = 0.2

class RandomMovesGate: 
    """Generate random moves for viability checking. --> keep if movement is significant"""

    def __init__(self, cfg: ViabilityConfig=ViabilityConfig()) -> None:
        self.cfg = cfg
    
    def controller(self, model: mj.MjModel, data: mj.MjData, robot) -> np.ndarray[np.float64]:
        nu = model.nu
        if nu == 0:
            return np.array([])  # No actuators to control
        t = float(data.time)
        if t < self.cfg.probe_settle_s:
            return np.zeros(nu, dtype=np.float64)  # No movement during settle time
        
        k = int((t - self.cfg.probe_settle_s) // self.cfg.probe_switch_s)
        rng = np.random.default_rng(self.cfg.probe_seed ^(nu << 16 ) ^ k)
        out = self.cfg.probe_amplitude * rng.uniform(-1.0, 1.0, size=nu)
        np.clip(out, -np.pi/2, np.pi/2, out=out)
        return out.astype(np.float64)
    
    def score(self, genotype: list[np.ndarray]) -> float:
        """Build body, run short probe, return distance moved."""
        try: 
            mj.set_mjcb_control(None)  
            robot = Robot(genotype)
            core = construct_mjspec_from_graph(robot.graph)
            tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
            ctrl = Controller(controller_callback_function=self.controller, tracker=tracker)
            experiment(robot=robot, core=core, controller=ctrl, environment=OlympicArena(), duration_s=self.cfg.duration_s)
            if tracker.history["xpos"] and len(tracker.history["xpos"][0]) > 1:
                pos = np.asarray(tracker.history["xpos"][0])
                return float(np.linalg.norm(pos[-1, :2]- pos[0,:2]))
            return 0.0
        finally:
            mj.set_mjcb_control(None)  

    def is_viable(self, genotype: list[np.ndarray]) -> bool:
        """Check if the genotype is viable based on movement score."""
        return self.score(genotype) >= self.cfg.treshold_distance                