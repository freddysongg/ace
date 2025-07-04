"""
Unit tests for the data_loader module.

This module contains test cases to verify the functionality of data loading,
chronological splitting, and preprocessing functions.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import (
    load_data,
    chronological_split,
    preprocess_data,
    get_data_info,
)


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data for testing
        # Format: [year, feature1, feature2, ..., ozone, pm25, no2]
        self.sample_data = np.array(
            [
                [2001, 10.1, 20.2, 30.3, 0.1, 0.2, 0.3],
                [2001, 11.1, 21.2, 31.3, 0.11, 0.21, 0.31],
                [2012, 12.1, 22.2, 32.3, 0.12, 0.22, 0.32],
                [2013, 13.1, 23.2, 33.3, 0.13, 0.23, 0.33],
                [2014, 14.1, 24.2, 34.3, 0.14, 0.24, 0.34],
                [2015, 15.1, 25.2, 35.3, 0.15, 0.25, 0.35],
            ]
        )

    def test_load_data_success(self):
        """Test successful data loading from .npy file."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
            np.save(tmp_file.name, self.sample_data)
            tmp_file.flush()

            loaded_data = load_data(tmp_file.name)

            np.testing.assert_array_equal(loaded_data, self.sample_data)

            # Clean up
            os.unlink(tmp_file.name)

    def test_load_data_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_data("non_existent_file.npy")

    def test_chronological_split_correct_splits(self):
        """Test that chronological split creates correct train/val/test splits."""
        train_data, val_data, test_data = chronological_split(
            self.sample_data, year_column_index=0
        )

        # Check that data is split correctly by year
        train_years = train_data[:, 0]
        val_years = val_data[:, 0]
        test_years = test_data[:, 0]

        # Training: 2001-2012
        self.assertTrue(np.all((train_years >= 2001) & (train_years <= 2012)))

        # Validation: 2013
        self.assertTrue(np.all(val_years == 2013))

        # Testing: 2014-2015
        self.assertTrue(np.all((test_years >= 2014) & (test_years <= 2015)))

        # Check expected shapes
        self.assertEqual(train_data.shape[0], 3)  # 2001, 2001, 2012
        self.assertEqual(val_data.shape[0], 1)  # 2013
        self.assertEqual(test_data.shape[0], 2)  # 2014, 2015

    def test_chronological_split_preserves_features(self):
        """Test that chronological split preserves all features."""
        train_data, val_data, test_data = chronological_split(self.sample_data)

        # Check that all splits have the same number of features
        expected_features = self.sample_data.shape[1]
        self.assertEqual(train_data.shape[1], expected_features)
        self.assertEqual(val_data.shape[1], expected_features)
        self.assertEqual(test_data.shape[1], expected_features)

    def test_preprocess_data_default_columns(self):
        """Test data preprocessing with default column assignments."""
        train_data, val_data, test_data = chronological_split(self.sample_data)
        processed = preprocess_data(train_data, val_data, test_data)

        # Check that all required keys are present
        required_keys = [
            "X_train",
            "y_train",
            "X_val",
            "y_val",
            "X_test",
            "y_test",
            "feature_columns",
            "target_columns",
        ]
        for key in required_keys:
            self.assertIn(key, processed)

        # Check shapes - assuming last 3 columns are targets
        expected_features = (
            self.sample_data.shape[1] - 3 - 1
        )  # Exclude year and 3 pollutants
        expected_targets = 3

        self.assertEqual(processed["X_train"].shape[1], expected_features)
        self.assertEqual(processed["y_train"].shape[1], expected_targets)

    def test_preprocess_data_custom_columns(self):
        """Test data preprocessing with custom column assignments."""
        train_data, val_data, test_data = chronological_split(self.sample_data)

        feature_cols = [1, 2, 3]  # Skip year column
        target_cols = [4, 5, 6]  # Last 3 columns

        processed = preprocess_data(
            train_data,
            val_data,
            test_data,
            feature_columns=feature_cols,
            target_columns=target_cols,
        )

        self.assertEqual(processed["X_train"].shape[1], len(feature_cols))
        self.assertEqual(processed["y_train"].shape[1], len(target_cols))

        # Verify column assignments are stored
        self.assertEqual(processed["feature_columns"], feature_cols)
        self.assertEqual(processed["target_columns"], target_cols)

    def test_get_data_info(self):
        """Test data information extraction."""
        info = get_data_info(self.sample_data)

        # Check that all required keys are present
        required_keys = [
            "shape",
            "dtype",
            "memory_usage_mb",
            "has_nan",
            "nan_count",
            "min_values",
            "max_values",
            "mean_values",
        ]
        for key in required_keys:
            self.assertIn(key, info)

        # Check specific values
        self.assertEqual(info["shape"], self.sample_data.shape)
        self.assertEqual(info["dtype"], self.sample_data.dtype)
        self.assertFalse(info["has_nan"])
        self.assertEqual(info["nan_count"], 0)

    def test_get_data_info_with_nan(self):
        """Test data information extraction with NaN values."""
        data_with_nan = self.sample_data.copy()
        data_with_nan[0, 1] = np.nan

        info = get_data_info(data_with_nan)

        self.assertTrue(info["has_nan"])
        self.assertEqual(info["nan_count"], 1)


class TestDataLoaderIntegration(unittest.TestCase):
    """Integration tests for the data loader module."""

    def test_full_pipeline(self):
        """Test the complete data loading and preprocessing pipeline."""
        # Create sample data
        sample_data = np.array(
            [
                [2001, 10.1, 20.2, 0.1, 0.2, 0.3],
                [2012, 12.1, 22.2, 0.12, 0.22, 0.32],
                [2013, 13.1, 23.2, 0.13, 0.23, 0.33],
                [2014, 14.1, 24.2, 0.14, 0.24, 0.34],
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
            np.save(tmp_file.name, sample_data)
            tmp_file.flush()

            try:
                # Load data
                data = load_data(tmp_file.name)

                # Split data
                train_data, val_data, test_data = chronological_split(data)

                # Preprocess data
                processed = preprocess_data(
                    train_data,
                    val_data,
                    test_data,
                    feature_columns=[1, 2],
                    target_columns=[3, 4, 5],
                )

                # Verify the complete pipeline works
                self.assertEqual(processed["X_train"].shape[1], 2)  # 2 features
                self.assertEqual(processed["y_train"].shape[1], 3)  # 3 targets

                # Verify data integrity
                self.assertTrue(processed["X_train"].shape[0] > 0)
                self.assertTrue(processed["X_val"].shape[0] > 0)
                self.assertTrue(processed["X_test"].shape[0] > 0)

            finally:
                # Clean up
                os.unlink(tmp_file.name)


if __name__ == "__main__":
    unittest.main()
