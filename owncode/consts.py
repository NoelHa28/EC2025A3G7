from pathlib import Path
import numpy as np
from typing import Literal

STOCHASTIC_SPAWN_POSITIONS: list[list[float]] = [
    [-0.8, 0, 0.1],  # start / flat
    [0.4,  0, 0.1],  # rugged
    [2.2,  0, 0.1],  # uphill
]

TARGET_POSITION = [5, 0, 0.5]

NUM_MODULES = 30

SEED = 1
RNG = np.random.default_rng()

GENOTYPE_SIZE = 64
GENOTYPE_RANGE = 10

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]