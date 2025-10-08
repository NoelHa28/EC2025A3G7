from nde import NeuralDevelopmentalEncodingWithLoading
from tqdm import tqdm
from robot import Robot
from morphology_constraints import is_robot_viable


import numpy as np


RNG = np.random.default_rng(42)
genotype_size = 64

genotype = [
    RNG.random(genotype_size).astype(np.float32),
    RNG.random(genotype_size).astype(np.float32),
    RNG.random(genotype_size).astype(np.float32),
]

robot = Robot(genotype)

