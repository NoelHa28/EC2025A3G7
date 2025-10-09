import numpy as np

STOCHASTIC_SPAWN_POSITIONS: list[list[float]] = [
    [-0.8, 0, 0.1],  # start / flat
    [0.4,  0, 0.1],  # rugged
    [2.2,  0, 0.1],  # uphill
]

TARGET_POSITION = [5, 0, 0.5]

NUM_MODULES = 25

SEED = 42
RNG = np.random.default_rng(SEED)

GENOTYPE_SIZE = 64