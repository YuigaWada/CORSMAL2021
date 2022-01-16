import random

import numpy
import torch

SEED = 0


def fix_random_seeds(specified_seed: int = SEED):
    random.seed(specified_seed)
    numpy.random.seed(seed=specified_seed)
    torch.manual_seed(seed=specified_seed)
    torch.cuda.manual_seed_all(seed=specified_seed)
