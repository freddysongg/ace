import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import (
    chronological_split,
    preprocess_data,
    create_lookback_sequences,
)


class TestDataPipelineValidation(unittest.TestCase):
    """Regression tests that lock down critical data-pipeline invariants."""

    def _build_sample_dataset(self) -> np.ndarray:
        """Create a tiny synthetic dataset covering train/val/test years.

        Layout per row: [year, f1, f2, ozone, pm25, no2]
        """
        rows = [
            [2001, 1.0, 2.0, 0.1, 0.2, 0.3],  # clean – train
            [2002, np.nan, 3.0, 0.1, 0.2, 0.3],  # NaN in features – train (filtered)
            [2003, 4.0, 5.0, 0.1, 0.2, 0.3],  # clean – train
            [2006, 13.0, 14.0, 0.1, 0.2, 0.3],  # clean – train
            [2013, 6.0, 7.0, 0.1, 0.2, 0.3],  # clean – val
            [2014, 8.0, np.inf, 0.1, 0.2, 0.3],  # Inf in features – test (filtered)
            [2014, 9.0, 10.0, 0.1, 0.2, 0.3],  # clean – test
        ]
        return np.asarray(rows, dtype=np.float64)


    def test_lookback_sequence_shape_has_multiple_features(self):
        data = self._build_sample_dataset()
        train, _, _ = chronological_split(data, year_column_index=0)

        processed = preprocess_data(
            train,
            train,  
            train,
            feature_columns=[1, 2],
            target_columns=[3, 4, 5],
        )

        X_seq, _ = create_lookback_sequences(
            processed["X_train"], processed["y_train"], lookback=2
        )

        self.assertGreater(
            X_seq.shape[2],
            1,
            msg="Expected >=2 features in look-back sequences but got {}".format(
                X_seq.shape[2]
            ),
        )

    def test_filter_finite_removes_non_finite_rows(self):
        data = self._build_sample_dataset()
        train, val, test = chronological_split(data, year_column_index=0)

        processed = preprocess_data(
            train,
            val,
            test,
            feature_columns=[1, 2],
            target_columns=[3, 4, 5],
        )

        for split in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
            arr = processed[split]
            self.assertTrue(
                np.isfinite(arr).all(),
                msg=f"Non-finite values found in {split} after preprocessing!",
            )

    def test_scaler_fitted_only_on_train_data(self):
        data = self._build_sample_dataset()
        train, val, test = chronological_split(data, year_column_index=0)

        processed = preprocess_data(
            train,
            val,
            test,
            feature_columns=[1, 2],
            target_columns=[3, 4, 5],
        )

        n_train_rows = processed["X_train"].shape[0]
        n_seen = processed["target_scaler"].n_samples_seen_
        if isinstance(n_seen, np.ndarray):
            self.assertTrue(
                np.all(n_seen == n_train_rows),
                msg="Scaler appears to have been fit on data outside the training set.",
            )
        else:
            self.assertEqual(
                n_seen,
                n_train_rows,
                msg="Scaler appears to have been fit on data outside the training set.",
            )


if __name__ == "__main__":
    unittest.main() 