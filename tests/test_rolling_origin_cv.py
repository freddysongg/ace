import unittest
import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import rolling_origin_splits


class TestRollingOriginCV(unittest.TestCase):
    """Unit tests for the rolling-origin cross-validation splitter."""

    @staticmethod
    def _years(arr: np.ndarray, year_col: int = 0):
        return np.unique(arr[:, year_col].astype(int))

    def test_generate_expected_year_windows(self):
        years = np.arange(2008, 2021)
        data = np.column_stack([years, np.zeros_like(years)])

        folds = list(
            rolling_origin_splits(
                data,
                year_column_index=0,
                initial_train_start=2009,
                train_window=4,
                n_folds=3,  
            )
        )

        train0, val0, test0 = folds[0]
        self.assertListEqual(self._years(train0).tolist(), list(range(2009, 2013)))
        self.assertListEqual(self._years(val0).tolist(), [2013])
        self.assertListEqual(self._years(test0).tolist(), [2014])

        train1, val1, test1 = folds[1]
        self.assertListEqual(self._years(train1).tolist(), list(range(2010, 2014)))
        self.assertListEqual(self._years(val1).tolist(), [2014])
        self.assertListEqual(self._years(test1).tolist(), [2015])

        train2, val2, test2 = folds[2]
        self.assertListEqual(self._years(train2).tolist(), list(range(2011, 2015)))
        self.assertListEqual(self._years(val2).tolist(), [2015])
        self.assertListEqual(self._years(test2).tolist(), [2016])

    def test_error_on_invalid_params(self):
        data = np.zeros((10, 2))
        with self.assertRaises(ValueError):
            list(
                rolling_origin_splits(
                    data, initial_train_start=2000, train_window=0, n_folds=1
                )
            )


if __name__ == "__main__":
    unittest.main() 