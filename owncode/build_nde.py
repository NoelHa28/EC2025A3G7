from nde import NeuralDevelopmentalEncodingWithLoading
from tqdm import tqdm
from robot import Robot
from morphology_constraints import is_robot_viable


import numpy as np


RNG = np.random.default_rng(42)
genotype_size = 64

robot = None
while robot is None or not is_robot_viable(robot, max_bricks_per_limb=3):
    genotype = [
        RNG.uniform(0, 1, genotype_size).astype(np.float32),
        RNG.uniform(-100, 100, genotype_size).astype(np.float32),
        RNG.uniform(-100, 100, genotype_size).astype(np.float32),
    ]
    robot = Robot(genotype)