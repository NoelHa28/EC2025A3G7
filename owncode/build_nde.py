from nde import NeuralDevelopmentalEncodingWithLoading
from tqdm import tqdm
from robot import Robot
from morphology_constraints import is_robot_viable


import numpy as np

from consts import RNG, GENOTYPE_SIZE

robot = None
while robot is None or not is_robot_viable(robot, max_bricks_per_limb=3):
    genotype = [
        RNG.uniform(-1, 1, GENOTYPE_SIZE).astype(np.float32),
        RNG.uniform(-1, 1, GENOTYPE_SIZE).astype(np.float32),
        RNG.uniform(-1, 1, GENOTYPE_SIZE).astype(np.float32),
    ]
    robot = Robot(body_genotype=genotype)