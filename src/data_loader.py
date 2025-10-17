"""
Data loading and preprocessing module for air pollutant prediction models.

This module contains functions to load data from .npy files and perform
chronological splits for training, validation, and testing.
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Iterator
from sklearn.preprocessing import StandardScaler, RobustScaler


# Load the statistically-determined column names if available
_suggested_names_path = Path("data") / "suggested_column_names.json"
if _suggested_names_path.exists():
    try:
        with open(_suggested_names_path, "r") as f:
            _DEFAULT_FIELD_NAMES = json.load(f)
    except Exception as e:
        print(
            f"Warning: Failed to load suggested column names. Using fallback. Error: {e}"
        )
        _DEFAULT_FIELD_NAMES = [
            f"col_{i}" for i in range(46)
        ]  # Fallback for 46 columns
else:
    # Fallback field names based on statistical analysis - these are the correct target positions
    _DEFAULT_FIELD_NAMES: List[str] = [
        "lon",  # 0
        "lat",  # 1
        "year",  # 2
        "month",  # 3
        "urban_fraction",  # 4
        "feature_5",  # 5
        "feature_6",  # 6
        "LCZ_0",  # 7
        "LCZ_17",  # 8
        "LCZ_12",  # 9
        "LCZ_11",  # 10
        "LCZ_14",  # 11
        "LCZ_15",  # 12
        "LCZ_13",  # 13
        "LCZ_16",  # 14
        "LCZ_6",  # 15
        "LCZ_8",  # 16
        "feature_16",  # 17
        "feature_17",  # 18
        "feature_18",  # 19
        "feature_19",  # 20
        "feature_20",  # 21
        "feature_21",  # 22
        "feature_22",  # 23
        "feature_23",  # 24
        "feature_24",  # 25
        "feature_25",  # 26
        "feature_26",  # 27
        "feature_27",  # 28
        "feature_28",  # 29
        "feature_29",  # 30
        "feature_30",  # 31
        "feature_31",  # 32
        "feature_32",  # 33
        "feature_33",  # 34
        "feature_34",  # 35
        "feature_35",  # 36
        "feature_36",  # 37
        "feature_37",  # 38
        "feature_38",  # 39
        "feature_39",  # 40
        "feature_40",  # 41
        "feature_41",  # 42
        "no2_concentration",  # 43 - Actual NO2 target
        "ozone",  # 44 - Actual ozone target
        "pm25_concentration",  # 45 - Actual PM2.5 target
    ]

_json_path = Path("data") / "final_column_names.json"

if _json_path.exists():
    try:
        with open(_json_path, "r") as _f:
            FIELD_NAMES: List[str] = json.load(_f)
    except Exception as _e:  # pragma: no cover – fallback path
        print(
            f"Warning: Failed to load column names from {_json_path}. Using default list. (Error: {_e})"
        )
        FIELD_NAMES = _DEFAULT_FIELD_NAMES
else:
    FIELD_NAMES = _DEFAULT_FIELD_NAMES

V2_FIELD_NAMES: List[str] = FIELD_NAMES


def load_data(file_path: str) -> np.ndarray:
    """
    Load data from a .npy file. It can handle both pre-processed 2D arrays
    and the original 1D structured arrays, which it will convert automatically.

    Args:
        file_path (str): Path to the .npy file.

    Returns:
        np.ndarray: Loaded 2D dataset.
    """
    try:
        data = np.load(file_path, allow_pickle=True)

        if data.ndim == 1 and data.dtype.names is not None:
            print("Loaded structured 1-D array – converting to 2-D matrix …")
            data = np.vstack([data[name] for name in data.dtype.names]).T
            print(f"Structured array converted to shape: {data.shape}")

        if data.ndim != 2:
            raise ValueError(
                f"Loaded data has an unsupported shape: {data.shape}. Expected a 2-D array."
            )

        print(f"Successfully loaded data with shape: {data.shape}")
        return data.astype(np.float32)

    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")


def chronological_split(
    data: np.ndarray,
    year_column_index: int = 0,
    *,
    train_years: Optional[List[int]] = None,
    val_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
    train_start: int = 2001,
    train_end: int = 2012,
    val_year: int = 2013,
    test_start: int = 2014,
    test_end: Optional[int] = 2015,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform chronological split of the dataset.

    The function supports two modes of operation:
    1. Using year lists (train_years, val_years, test_years)
    2. Using start/end ranges (train_start/end, val_year, test_start/end)

    If year lists are provided, they take precedence over the start/end parameters.

    Args:
        data (np.ndarray): Input dataset
        year_column_index (int): Index of the column containing year information
        train_years (List[int], optional): List of years to include in training set
        val_years (List[int], optional): List of years to include in validation set
        test_years (List[int], optional): List of years to include in test set
        train_start (int): Start year for training (default: 2001)
        train_end (int): End year for training (default: 2012)
        val_year (int): Year for validation (default: 2013)
        test_start (int): Start year for testing (default: 2014)
        test_end (Optional[int]): End year for testing (default: 2015)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Training, validation, and test sets

    Raises:
        ValueError: If train_years is None when using year list mode
        ValueError: If year_column_index is out of bounds
        ValueError: If all resulting splits are empty
    """

    def _infer_year_col(arr: np.ndarray, sample_size: int = 10000) -> int:
        """Heuristically detect the year column by searching for 4-digit integers."""
        n_cols = arr.shape[1]
        sample = arr[:sample_size] if arr.shape[0] > sample_size else arr

        for col in range(n_cols):
            col_vals = sample[:, col]
            if not np.issubdtype(col_vals.dtype, np.number):
                continue
            years_int = col_vals.astype(int)
            if (
                np.all((years_int >= 1900) & (years_int <= 2100))
                and len(np.unique(years_int)) > 5
            ):
                return col
        return 0

    years = data[:, year_column_index]

    # Determine which mode to use based on provided parameters
    use_year_lists = any(x is not None for x in [train_years, val_years, test_years])

    if use_year_lists:
        # Validate year lists
        if train_years is None:
            raise ValueError("train_years must be provided when using year list mode")

        # Validate that year lists contain valid integers
        if not all(isinstance(y, (int, np.integer)) for y in train_years):
            raise ValueError("train_years must contain only integer values")

        if val_years is not None and not all(
            isinstance(y, (int, np.integer)) for y in val_years
        ):
            raise ValueError("val_years must contain only integer values")

        if test_years is not None and not all(
            isinstance(y, (int, np.integer)) for y in test_years
        ):
            raise ValueError("test_years must contain only integer values")

        # Check for overlapping years between splits
        if val_years is not None and any(y in train_years for y in val_years):
            print("Warning: Some years appear in both train_years and val_years")

        if test_years is not None and any(y in train_years for y in test_years):
            print("Warning: Some years appear in both train_years and test_years")

        if (
            val_years is not None
            and test_years is not None
            and any(y in val_years for y in test_years)
        ):
            print("Warning: Some years appear in both val_years and test_years")

        # Create masks using np.isin() for each dataset
        train_mask = np.isin(years, train_years)

        # For validation and test, use empty lists if None is provided
        val_mask = np.isin(years, val_years if val_years is not None else [])
        test_mask = np.isin(years, test_years if test_years is not None else [])

        # Log the years used for each split
        print(f"Using year lists mode:")
        print(f"  Training years: {train_years}")
        print(f"  Validation years: {val_years if val_years is not None else []}")
        print(f"  Test years: {test_years if test_years is not None else []}")
    else:
        # Use traditional start/end parameters
        train_mask = (years >= train_start) & (years <= train_end)
        val_mask = years == val_year
        if test_end is None:
            test_mask = years >= test_start
        else:
            test_mask = (years >= test_start) & (years <= test_end)

    # Check if any of the masks are empty, and try to infer the year column if so
    if train_mask.sum() == 0 and val_mask.sum() == 0 and test_mask.sum() == 0:
        inferred_col = _infer_year_col(data)
        if inferred_col != year_column_index:
            print(
                f"Year column index {year_column_index} produced empty splits. "
                f"Auto-detected year column {inferred_col}. Re-splitting…"
            )
            year_column_index = inferred_col
            years = data[:, year_column_index]

            # Reapply the masks with the inferred year column
            if use_year_lists:
                train_mask = np.isin(years, train_years)
                val_mask = np.isin(years, val_years if val_years is not None else [])
                test_mask = np.isin(years, test_years if test_years is not None else [])
            else:
                train_mask = (years >= train_start) & (years <= train_end)
                val_mask = years == val_year
                if test_end is None:
                    test_mask = years >= test_start
                else:
                    test_mask = (years >= test_start) & (years <= test_end)

    # Apply masks to get the data splits
    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    # Validate that we have data in each split
    if train_data.shape[0] == 0:
        print("Warning: Training set is empty after splitting")
    if val_data.shape[0] == 0:
        print("Warning: Validation set is empty after splitting")
    if test_data.shape[0] == 0:
        print("Warning: Test set is empty after splitting")

    # Raise error if all splits are empty
    if train_data.shape[0] == 0 and val_data.shape[0] == 0 and test_data.shape[0] == 0:
        available_years = np.unique(years)
        raise ValueError(
            f"All data splits are empty. Check your year parameters. "
            f"Available years in the dataset: {available_years}"
        )

    # Warn if training set is empty but other sets have data
    if train_data.shape[0] == 0 and (val_data.shape[0] > 0 or test_data.shape[0] > 0):
        print("ERROR: Training set is empty but validation or test sets have data.")
        print("This will likely cause model training to fail.")

    print(f"Training set shape: {train_data.shape}")
    print(f"Validation set shape: {val_data.shape}")
    print(f"Test set shape: {test_data.shape}")

    return train_data, val_data, test_data


def random_split(
    data: np.ndarray,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle *data* once and return train / val / test partitions.

    Parameters
    ----------
    data
        2-D *(N, F)* matrix with samples along axis 0.
    train_frac
        Fraction of rows to allocate to the training set. Default 0.8.
    val_frac
        Fraction for validation. The remainder goes to the test set.
    seed
        Random seed for reproducibility.

    Returns
    -------
    train, val, test : np.ndarray
    """

    if not (0 < train_frac < 1) or not (0 <= val_frac < 1):
        raise ValueError("train_frac and val_frac must be within (0,1)")

    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1 – leave room for test set")

    N = data.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    train = data[train_idx]
    val = data[val_idx]
    test = data[test_idx]

    print(
        f"Random split with seed {seed}: train {train.shape}, val {val.shape}, test {test.shape}"
    )

    return train, val, test


def preprocess_data(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    feature_columns: list = None,
    target_columns: list = None,
    target_column_index: Optional[int] = None,
    truncate_target_percentile: Optional[float] = None,
    lat_col_name: str = "lat",
    lon_col_name: str = "lon",
    log_transform_targets: Optional[list] = None,
    use_robust_scaler_targets: bool = False,
) -> Dict[str, Any]:
    """
    Preprocess the data by separating features and targets.

    Args:
        train_data (np.ndarray): Training dataset
        val_data (np.ndarray): Validation dataset
        test_data (np.ndarray): Test dataset
        feature_columns (list): Indices of feature columns
        target_columns (list): Indices of target columns (pollutants)
        target_column_index (int, optional): Index within target_columns to process
            as single pollutant. If provided, only this target will be processed,
            resulting in 1D target arrays. Maintains backward compatibility when None.
        truncate_target_percentile (float, optional): If provided, removes rows where
            target values are below this percentile threshold. Threshold is calculated
            on training data only and applied to all datasets consistently.
        lat_col_name (str): Column name for latitude (used if *FIELD_NAMES* is available).
        lon_col_name (str): Column name for longitude (used if *FIELD_NAMES* is available).
        log_transform_targets (list, optional): Indices (relative to *target_columns*)
            of targets to which a natural log1p transform should be applied prior
            to scaling.  Useful for correcting heavy-tailed distributions (e.g.
            Ozone and NO2).
        use_robust_scaler_targets (bool): If True, uses sklearn's RobustScaler for
            the target variables instead of StandardScaler. Helpful for heavy-
            tailed targets like PM2.5 without applying log-transforms.

    Returns:
        Dict[str, Any]: Dictionary containing processed features and targets
    """
    if feature_columns is None:
        feature_columns = list(range(1, train_data.shape[1] - 3))

    if target_columns is None:
        target_columns = list(range(train_data.shape[1] - 3, train_data.shape[1]))

    processed_data = {
        "feature_scaler": None,
        "target_scaler": None,
    }

    feature_scaler = StandardScaler()
    target_scaler = RobustScaler() if use_robust_scaler_targets else StandardScaler()

    X_train_raw = train_data[:, feature_columns]
    y_train_raw = train_data[:, target_columns]

    if target_column_index is not None:
        if target_column_index < 0 or target_column_index >= len(target_columns):
            raise IndexError(
                f"target_column_index {target_column_index} is out of bounds for target_columns of length {len(target_columns)}"
            )
        print(f"Processing single pollutant at target index {target_column_index}")
        y_train_raw = y_train_raw[:, target_column_index : target_column_index + 1]

    def _filter_finite(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
        if not mask.all():
            n_removed = (~mask).sum()
            print(f"Filtering out {n_removed} rows with non-finite values …")
        return X[mask], y[mask]

    X_train_raw, y_train_raw = _filter_finite(X_train_raw, y_train_raw)

    X_val_raw = val_data[:, feature_columns]
    y_val_raw = val_data[:, target_columns]

    if target_column_index is not None:
        y_val_raw = y_val_raw[:, target_column_index : target_column_index + 1]

    X_val_raw, y_val_raw = _filter_finite(X_val_raw, y_val_raw)

    X_test_raw = test_data[:, feature_columns]
    y_test_raw = test_data[:, target_columns]

    if target_column_index is not None:
        y_test_raw = y_test_raw[:, target_column_index : target_column_index + 1]

    X_test_raw, y_test_raw = _filter_finite(X_test_raw, y_test_raw)

    # Apply target truncation if specified
    truncation_applied = False
    truncation_threshold = None
    rows_removed = {"train": 0, "val": 0, "test": 0}

    if truncate_target_percentile is not None:
        # Validate percentile range
        if not isinstance(truncate_target_percentile, (int, float, np.number)):
            raise ValueError(
                f"truncate_target_percentile must be a numeric value, got {type(truncate_target_percentile)}"
            )

        if not (0 <= truncate_target_percentile <= 100):
            raise ValueError(
                f"truncate_target_percentile must be between 0 and 100, got {truncate_target_percentile}"
            )

        # Calculate threshold on training data only
        if target_column_index is not None:
            # Single target case
            if y_train_raw.shape[0] == 0:
                raise ValueError(
                    "Cannot calculate percentile threshold: training data is empty"
                )

            truncation_threshold = np.percentile(
                y_train_raw[:, 0], truncate_target_percentile
            )

            # Apply truncation to all datasets
            train_mask = y_train_raw[:, 0] >= truncation_threshold
            val_mask = y_val_raw[:, 0] >= truncation_threshold
            test_mask = y_test_raw[:, 0] >= truncation_threshold
        else:
            # Multiple targets case - use first target for truncation
            if y_train_raw.shape[0] == 0:
                raise ValueError(
                    "Cannot calculate percentile threshold: training data is empty"
                )

            truncation_threshold = np.percentile(
                y_train_raw[:, 0], truncate_target_percentile
            )

            # Apply truncation to all datasets
            train_mask = y_train_raw[:, 0] >= truncation_threshold
            val_mask = y_val_raw[:, 0] >= truncation_threshold
            test_mask = y_test_raw[:, 0] >= truncation_threshold

        # Count rows to be removed
        rows_removed["train"] = (~train_mask).sum()
        rows_removed["val"] = (~val_mask).sum()
        rows_removed["test"] = (~test_mask).sum()

        # Check if truncation would remove too much data
        total_train = y_train_raw.shape[0]
        total_val = y_val_raw.shape[0]
        total_test = y_test_raw.shape[0]

        train_removal_pct = (
            (rows_removed["train"] / total_train * 100) if total_train > 0 else 0
        )
        val_removal_pct = (
            (rows_removed["val"] / total_val * 100) if total_val > 0 else 0
        )
        test_removal_pct = (
            (rows_removed["test"] / total_test * 100) if total_test > 0 else 0
        )

        # Warn if truncation would remove more than 50% of any dataset
        if train_removal_pct > 50:
            print(
                f"Warning: Truncation will remove {train_removal_pct:.1f}% of training data"
            )
        if val_removal_pct > 50:
            print(
                f"Warning: Truncation will remove {val_removal_pct:.1f}% of validation data"
            )
        if test_removal_pct > 50:
            print(
                f"Warning: Truncation will remove {test_removal_pct:.1f}% of test data"
            )

        # Apply masks to filter data
        X_train_raw = X_train_raw[train_mask]
        y_train_raw = y_train_raw[train_mask]
        X_val_raw = X_val_raw[val_mask]
        y_val_raw = y_val_raw[val_mask]
        X_test_raw = X_test_raw[test_mask]
        y_test_raw = y_test_raw[test_mask]

        truncation_applied = True

        # Log truncation statistics
        print(f"Target truncation applied:")
        print(f"  Percentile threshold: {truncate_target_percentile}%")
        print(f"  Threshold value: {truncation_threshold:.4f}")
        print(
            f"  Rows removed - Train: {rows_removed['train']} ({train_removal_pct:.1f}%), "
            f"Val: {rows_removed['val']} ({val_removal_pct:.1f}%), "
            f"Test: {rows_removed['test']} ({test_removal_pct:.1f}%)"
        )
        print(
            f"  Remaining data - Train: {X_train_raw.shape[0]}, Val: {X_val_raw.shape[0]}, Test: {X_test_raw.shape[0]}"
        )

        # Ensure sufficient data remains
        min_required_samples = 10  # Minimum number of samples required for each set

        if X_train_raw.shape[0] < min_required_samples:
            raise ValueError(
                f"Insufficient training data remains after {truncate_target_percentile}% truncation "
                f"(got {X_train_raw.shape[0]} samples, need at least {min_required_samples})"
            )
        if X_val_raw.shape[0] < min_required_samples:
            raise ValueError(
                f"Insufficient validation data remains after {truncate_target_percentile}% truncation "
                f"(got {X_val_raw.shape[0]} samples, need at least {min_required_samples})"
            )
        if X_test_raw.shape[0] < min_required_samples:
            raise ValueError(
                f"Insufficient test data remains after {truncate_target_percentile}% truncation "
                f"(got {X_test_raw.shape[0]} samples, need at least {min_required_samples})"
            )

    if log_transform_targets:
        print(
            f"Applying np.log1p transformation to target columns (relative indices): {log_transform_targets}"
        )

        for tgt_idx in log_transform_targets:
            if target_column_index is not None:
                if tgt_idx == target_column_index:
                    y_train_raw[:, 0] = np.log1p(np.maximum(0, y_train_raw[:, 0]))
                    y_val_raw[:, 0] = np.log1p(np.maximum(0, y_val_raw[:, 0]))
                    y_test_raw[:, 0] = np.log1p(np.maximum(0, y_test_raw[:, 0]))
            else:
                if tgt_idx < 0 or tgt_idx >= len(target_columns):
                    raise IndexError(
                        f"Target index {tgt_idx} is out of bounds for target_columns of length {len(target_columns)}"
                    )

                y_train_raw[:, tgt_idx] = np.log1p(
                    np.maximum(0, y_train_raw[:, tgt_idx])
                )
                y_val_raw[:, tgt_idx] = np.log1p(np.maximum(0, y_val_raw[:, tgt_idx]))
                y_test_raw[:, tgt_idx] = np.log1p(np.maximum(0, y_test_raw[:, tgt_idx]))

        X_train_raw, y_train_raw = _filter_finite(X_train_raw, y_train_raw)
        X_val_raw, y_val_raw = _filter_finite(X_val_raw, y_val_raw)
        X_test_raw, y_test_raw = _filter_finite(X_test_raw, y_test_raw)

    X_train_scaled = feature_scaler.fit_transform(X_train_raw)
    y_train_scaled = target_scaler.fit_transform(y_train_raw)

    X_val_scaled = feature_scaler.transform(X_val_raw)
    y_val_scaled = target_scaler.transform(y_val_raw)

    X_test_scaled = feature_scaler.transform(X_test_raw)
    y_test_scaled = target_scaler.transform(y_test_raw)

    # Get feature names for the selected feature columns
    feature_names = [FIELD_NAMES[i] for i in feature_columns if i < len(FIELD_NAMES)]

    processed_data.update(
        {
            "X_train": X_train_scaled,
            "y_train": y_train_scaled,
            "X_val": X_val_scaled,
            "y_val": y_val_scaled,
            "X_test": X_test_scaled,
            "y_test": y_test_scaled,
            "X_train_raw": X_train_raw,
            "y_train_raw": y_train_raw,
            "X_val_raw": X_val_raw,
            "y_val_raw": y_val_raw,
            "X_test_raw": X_test_raw,
            "y_test_raw": y_test_raw,
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "target_column_index": target_column_index,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "log_transform_targets": log_transform_targets,
            "use_robust_scaler_targets": use_robust_scaler_targets,
            # Feature metadata
            "feature_names": feature_names,
            "column_names": FIELD_NAMES,
            "feature_indices": feature_columns,
            # Truncation metadata
            "truncation_applied": truncation_applied,
            "truncation_threshold": truncation_threshold,
            "rows_removed": rows_removed,
        }
    )

    if lat_col_name in V2_FIELD_NAMES and lon_col_name in V2_FIELD_NAMES:
        lat_col_idx = V2_FIELD_NAMES.index(lat_col_name)
        lon_col_idx = V2_FIELD_NAMES.index(lon_col_name)

        if target_column_index is not None:
            target_cols_for_filtering = [target_columns[target_column_index]]
        else:
            target_cols_for_filtering = target_columns

        mask_test = np.isfinite(test_data[:, feature_columns]).all(
            axis=1
        ) & np.isfinite(test_data[:, target_cols_for_filtering]).all(axis=1)

        processed_data["lats_test"] = test_data[mask_test, lat_col_idx]
        processed_data["lons_test"] = test_data[mask_test, lon_col_idx]

        print(
            "Extracted latitude and longitude columns for spatial plotting (filtered to finite rows)."
        )

    print(f"Features (scaled) shape: {processed_data['X_train'].shape}")
    print(f"Targets (scaled) shape: {processed_data['y_train'].shape}")

    return processed_data


def get_data_info(data: np.ndarray) -> Dict[str, Any]:
    """
    Get basic information about the dataset.

    Args:
        data (np.ndarray): Input dataset

    Returns:
        Dict[str, Any]: Dictionary containing dataset information
    """
    info = {
        "shape": data.shape,
        "dtype": data.dtype,
        "memory_usage_mb": data.nbytes / (1024 * 1024),
        "has_nan": np.isnan(data).any(),
        "nan_count": np.isnan(data).sum(),
        "min_values": np.nanmin(data, axis=0),
        "max_values": np.nanmax(data, axis=0),
        "mean_values": np.nanmean(data, axis=0),
    }

    return info


def get_field_names() -> Optional[List[str]]:
    """Returns the canonical field names for the dataset."""
    return FIELD_NAMES


def create_lookback_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int = 7,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate supervised sequences for sequence models.

    Parameters
    ----------
    features
        2-D array *(N, F)* after scaling / preprocessing.
    targets
        2-D array *(N, T)* (or 1-D *(N,)*). Must align with *features* in the first
        dimension.
    lookback
        Number of past timesteps to include in each sequence.
    step
        Stride between successive sequences. ``1`` keeps every possible sample.

    Returns
    -------
    X_seq
        3-D array *(M, lookback, F)*.
    y_seq
        2-D array *(M, T)*.
    """

    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must align along axis 0")

    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    N, F = features.shape
    T = targets.shape[1]

    num_seqs = (N - lookback) // step
    if num_seqs <= 0:
        raise ValueError(
            f"Not enough rows ({N}) for lookback window {lookback} with step {step}."
        )

    seqs = []
    labels = []
    for end in range(lookback, N, step):
        seqs.append(features[end - lookback : end])
        labels.append(targets[end])

    X_seq = np.stack(seqs)
    y_seq = np.stack(labels)

    return X_seq.astype(features.dtype), y_seq.astype(targets.dtype)


def regional_split(
    data: np.ndarray,
    test_region_bbox: Tuple[
        float, float, float, float
    ],  # (min_lon, min_lat, max_lon, max_lat)
    lat_col_name: str = "lat",
    lon_col_name: str = "lon",
    train_val_split_method: str = "chronological",  # or "random"
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
    **split_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data based on geographic region, using a bounding box to define the test set.

    Args:
        data (np.ndarray): Input dataset
        test_region_bbox (Tuple[float, float, float, float]): Bounding box for test region
                                                             (min_lon, min_lat, max_lon, max_lat)
        lat_col_name (str): Column name for latitude
        lon_col_name (str): Column name for longitude
        train_val_split_method (str): Method to split remaining data into train/val sets
                                     ("chronological" or "random")
        train_frac (float): Fraction of non-test data for training (used with random split)
        val_frac (float): Fraction of non-test data for validation (used with random split)
        seed (int): Random seed for reproducibility
        **split_kwargs: Additional arguments to pass to the train/val split method

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Training, validation, and test sets
    """
    # Validate the bounding box
    if len(test_region_bbox) != 4:
        raise ValueError(
            f"test_region_bbox must be a tuple of 4 values (min_lon, min_lat, max_lon, max_lat), "
            f"got {len(test_region_bbox)} values"
        )

    min_lon, min_lat, max_lon, max_lat = test_region_bbox

    # Check that all values are numeric
    if not all(isinstance(x, (int, float, np.number)) for x in test_region_bbox):
        raise ValueError("All bounding box coordinates must be numeric values")

    # Check that min values are less than max values
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(
            f"Invalid bounding box: min_lon ({min_lon}) must be less than max_lon ({max_lon}) "
            f"and min_lat ({min_lat}) must be less than max_lat ({max_lat})"
        )

    # Check that longitude values are within reasonable range (-180 to 180)
    if min_lon < -180 or max_lon > 180:
        print(
            f"Warning: Longitude values ({min_lon}, {max_lon}) outside typical range (-180 to 180)"
        )

    # Check that latitude values are within reasonable range (-90 to 90)
    if min_lat < -90 or max_lat > 90:
        print(
            f"Warning: Latitude values ({min_lat}, {max_lat}) outside typical range (-90 to 90)"
        )

    # Identify latitude and longitude column indices
    try:
        if FIELD_NAMES:
            if lat_col_name in FIELD_NAMES and lon_col_name in FIELD_NAMES:
                lat_col_idx = FIELD_NAMES.index(lat_col_name)
                lon_col_idx = FIELD_NAMES.index(lon_col_name)
                print(
                    f"Found lat column at index {lat_col_idx} and lon column at index {lon_col_idx}"
                )
            else:
                raise ValueError(
                    f"Column names {lat_col_name} and/or {lon_col_name} not found in FIELD_NAMES"
                )
        else:
            raise ValueError("FIELD_NAMES not available")
    except (ValueError, NameError):
        # Fallback: try to infer lat/lon columns by looking for typical ranges
        print(
            "Warning: Could not find lat/lon columns by name. Attempting to infer by value ranges..."
        )
        lat_col_idx = None
        lon_col_idx = None

        for i in range(data.shape[1]):
            col_data = data[:, i]
            col_min, col_max = np.nanmin(col_data), np.nanmax(col_data)

            # Latitude typically ranges from -90 to 90
            if -90 <= col_min <= 90 and -90 <= col_max <= 90 and col_max - col_min > 1:
                if lat_col_idx is None:
                    lat_col_idx = i
                    print(
                        f"Inferred lat column at index {lat_col_idx} (range: {col_min:.2f} to {col_max:.2f})"
                    )

            # Longitude typically ranges from -180 to 180
            if (
                -180 <= col_min <= 180
                and -180 <= col_max <= 180
                and col_max - col_min > 1
            ):
                if lon_col_idx is None:
                    lon_col_idx = i
                    print(
                        f"Inferred lon column at index {lon_col_idx} (range: {col_min:.2f} to {col_max:.2f})"
                    )

        if lat_col_idx is None or lon_col_idx is None:
            raise ValueError("Could not identify latitude and longitude columns")

    # Create boolean mask for data points within the test region bounding box
    lats = data[:, lat_col_idx]
    lons = data[:, lon_col_idx]

    test_mask = (
        (lons >= min_lon) & (lons <= max_lon) & (lats >= min_lat) & (lats <= max_lat)
    )

    # Extract test set
    test_data = data[test_mask]

    # Extract remaining data (not in test region)
    remaining_data = data[~test_mask]

    # Check if we have sufficient data
    if test_data.shape[0] == 0:
        # Provide more helpful error message with data bounds
        if lats.size > 0 and lons.size > 0:
            lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)
            lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
            raise ValueError(
                f"No data points found within the specified test region bounding box: "
                f"({min_lon}, {min_lat}, {max_lon}, {max_lat}). "
                f"Available data range: lat [{lat_min:.4f}, {lat_max:.4f}], "
                f"lon [{lon_min:.4f}, {lon_max:.4f}]"
            )
        else:
            raise ValueError(
                "No data points found within the specified test region bounding box"
            )

    if remaining_data.shape[0] == 0:
        raise ValueError(
            "No data points found outside the specified test region bounding box. "
            "The entire dataset falls within the test region."
        )

    # Check if test region contains very few points (less than 5% of the data)
    total_points = data.shape[0]
    test_points = test_data.shape[0]
    test_percentage = (test_points / total_points) * 100

    if test_percentage < 5:
        print(
            f"Warning: Test region contains only {test_percentage:.1f}% of the data "
            f"({test_points} out of {total_points} points)"
        )

    print(
        f"Regional split: {test_data.shape[0]} points in test region, {remaining_data.shape[0]} points outside"
    )

    # Split remaining data into train and validation sets
    if train_val_split_method.lower() == "chronological":
        # Use chronological split for train/val
        train_data, val_data, _ = chronological_split(remaining_data, **split_kwargs)
    elif train_val_split_method.lower() == "random":
        # For random split, we need to handle the case differently since we're only splitting into train/val
        # and not creating a test set from the remaining data
        if train_frac + val_frac > 1.0:
            # Adjust fractions to sum to 1.0
            adjusted_train_frac = train_frac / (train_frac + val_frac)
            adjusted_val_frac = val_frac / (train_frac + val_frac)
            print(f"Adjusted train_frac from {train_frac} to {adjusted_train_frac}")
            print(f"Adjusted val_frac from {val_frac} to {adjusted_val_frac}")
            train_frac = adjusted_train_frac
            val_frac = adjusted_val_frac

        N = remaining_data.shape[0]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)

        n_train = int(N * train_frac)
        n_val = N - n_train  # Use all remaining data for validation

        train_idx = idx[:n_train]
        val_idx = idx[n_train:]

        train_data = remaining_data[train_idx]
        val_data = remaining_data[val_idx]

        print(
            f"Random split with seed {seed}: train {train_data.shape}, val {val_data.shape}"
        )
    else:
        raise ValueError(
            f"Unsupported train_val_split_method: {train_val_split_method}. "
            f"Use 'chronological' or 'random'."
        )

    # Print split information
    print(f"Training set shape: {train_data.shape}")
    print(f"Validation set shape: {val_data.shape}")
    print(f"Test set shape: {test_data.shape}")

    return train_data, val_data, test_data


def rolling_origin_splits(
    data: np.ndarray,
    year_column_index: int = 0,
    *,
    initial_train_start: int,
    train_window: int,
    n_folds: int,
    val_window: int = 1,
    test_window: int = 1,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if train_window < 1 or val_window < 1 or n_folds < 1:
        raise ValueError("train_window, val_window and n_folds must all be >= 1")

    for k in range(n_folds):
        train_start = initial_train_start + k
        train_end = train_start + train_window - 1
        val_year_start = train_end + 1
        val_year_end = val_year_start + val_window - 1

        if val_window == 1:
            val_year = val_year_start
        else:
            val_year = val_year_start

        if test_window == 0:
            test_start = val_year_end + 1
            test_end = val_year_end
        else:
            test_start = val_year_end + 1
            test_end = test_start + test_window - 1

        train, val, test = chronological_split(
            data,
            year_column_index=year_column_index,
            train_start=train_start,
            train_end=train_end,
            val_year=val_year,
            test_start=test_start,
            test_end=test_end,
        )

        yield train, val, test
