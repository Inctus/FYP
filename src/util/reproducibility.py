import torch
import numpy as np

from constants import RANDOM_SEED

import random

def make_reproducible():
    """
    Sets the random seed for reproducibility across all libraries used
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False