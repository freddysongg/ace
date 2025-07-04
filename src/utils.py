import os
import random
from typing import Optional

import numpy as np
import tensorflow as tf

__all__ = ["set_global_seed"]


def set_global_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and TensorFlow PRNG seeds for reproducibility.

    Args:
        seed: Any non-negative integer. Using the same value across runs makes
            them (mostly) deterministic and comparable.
    """
    if seed < 0:
        raise ValueError("Seed must be a non-negative integer.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
