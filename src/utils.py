import os
import random
from typing import Optional

import numpy as np
import tensorflow as tf

__all__ = ["set_global_seed"]


def set_global_seed(seed: int = 42) -> None:
    """
    Configure reproducible random seeds across all libraries.
    
    Sets seeds for Python, NumPy, TensorFlow, and system hash functions
    to ensure consistent results across training runs.
    """
    if seed < 0:
        raise ValueError("Seed must be a non-negative integer.")

    # Set seeds for all random number generators
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python hash function seed
    random.seed(seed)                         # Python random module
    np.random.seed(seed)                     # NumPy random functions
    tf.random.set_seed(seed)                 # TensorFlow operations
