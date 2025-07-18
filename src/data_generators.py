"""Memory-efficient data generators for large datasets."""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional


class SequenceDataGenerator(tf.keras.utils.Sequence):
    """Memory-efficient generator for creating lookback sequences on-the-fly.
    
    This generator creates sequences during training to avoid loading all
    sequences into memory at once.
    """
    
    def __init__(self, 
                 X: np.ndarray,
                 y: np.ndarray,
                 lookback: int = 7,
                 step: int = 1,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """
        Initialize the sequence generator.
        
        Args:
            X: Input features array (samples, features)
            y: Target array (samples, targets)
            lookback: Number of timesteps to look back
            step: Step size for sequences
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
        """
        self.X = X
        self.y = y
        self.lookback = lookback
        self.step = step
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate valid indices for sequence generation
        self.valid_indices = np.arange(lookback, len(X), step)
        self.n_samples = len(self.valid_indices)
        
        # Initialize indices
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data."""
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Generate sequences for this batch
        actual_batch_size = len(batch_indices)
        X_batch = np.empty((actual_batch_size, self.lookback, self.X.shape[1]), dtype=np.float32)
        y_batch = np.empty((actual_batch_size, self.y.shape[1]), dtype=np.float32)
        
        for i, idx in enumerate(batch_indices):
            # Get the actual data index
            data_idx = self.valid_indices[idx]
            
            # Create sequence
            X_batch[i] = self.X[data_idx - self.lookback:data_idx]
            y_batch[i] = self.y[data_idx]
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        """Update indices after each epoch."""
        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

