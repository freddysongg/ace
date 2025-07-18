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


_DEFAULT_FIELD_NAMES: List[str] = [
    "lon",
    "lat",
    "year",
    "ozone",
    "urban_fraction",
    "road_density_m",
    "LCZ_0",
    "LCZ_17",
    "LCZ_12",
    "LCZ_11",
    "LCZ_14",
    "LCZ_15",
    "LCZ_13",
    "LCZ_16",
    "LCZ_6",
    "LCZ_8",
    "tasmax_monmean",
    "pr_monmean",
    "huss_monmean",
    "rsds_monmean",
    "uas_monmean",
    "vas_monmean",
    "windspeed_monmean",
    "tasmax_minus_tasmin_monmean",
    "pm25_concentration",
    "no2_concentration",
    "population",
    "elevation",
    "distance_to_coast_km",
    "oc_x_tasmax",
    "bc_x_windspeed",
    "nox_x_rhmin",
    "urbanfrac_x_coastdist",
    "road_dense_x_LCZ13",
    "tasmax_x_pr",
    "elev_x_lat",
    "so2_x_rsds",
    "pollutant_pc1",
    "pollutant_pc2",
    "pollutant_pc3",
    "LCZ_dense_urban",
    "rel_humid_mean_monmean",
    "rel_humid_range_monmean",
    "month_sin",
    "month_cos",
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
    train_start: int = 2001,
    train_end: int = 2012,
    val_year: int = 2013,
    test_start: int = 2014,
    test_end: Optional[int] = 2015,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform chronological split of the dataset.

    Training: 2001-2012
    (default) Training: 2001-2012
    Validation: 2013
    Testing: 2014-2015

    You can override these year ranges via keyword arguments (*train_start*,
    *train_end*, *val_year*, *test_start*, *test_end*) to implement a sliding
    window or other custom splits. Set *test_end* to ``None`` to include all
    years ≥ *test_start* in the test set.

    Args:
        data (np.ndarray): Input dataset
        year_column_index (int): Index of the column containing year information

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Training, validation, and test sets
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

    train_mask = (years >= train_start) & (years <= train_end)
    val_mask = years == val_year
    if test_end is None:
        test_mask = years >= test_start
    else:
        test_mask = (years >= test_start) & (years <= test_end)

    if train_mask.sum() == 0 and val_mask.sum() == 0 and test_mask.sum() == 0:
        inferred_col = _infer_year_col(data)
        if inferred_col != year_column_index:
            print(
                f"Year column index {year_column_index} produced empty splits. "
                f"Auto-detected year column {inferred_col}. Re-splitting…"
            )
            year_column_index = inferred_col
            years = data[:, year_column_index]
            train_mask = (years >= train_start) & (years <= train_end)
            val_mask = years == val_year
            if test_end is None:
                test_mask = years >= test_start
            else:
                test_mask = (years >= test_start) & (years <= test_end)

    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

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

    # Handle single-pollutant processing
    if target_column_index is not None:
        if target_column_index < 0 or target_column_index >= len(target_columns):
            raise IndexError(
                f"target_column_index {target_column_index} is out of bounds for target_columns of length {len(target_columns)}"
            )
        print(f"Processing single pollutant at target index {target_column_index}")
        # Extract single target column, keeping as 2D for compatibility with existing logic
        y_train_raw = y_train_raw[:, target_column_index:target_column_index+1]

    def _filter_finite(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
        if not mask.all():
            n_removed = (~mask).sum()
            print(f"Filtering out {n_removed} rows with non-finite values …")
        return X[mask], y[mask]

    X_train_raw, y_train_raw = _filter_finite(X_train_raw, y_train_raw)

    X_val_raw = val_data[:, feature_columns]
    y_val_raw = val_data[:, target_columns]
    
    # Apply single-pollutant processing to validation data
    if target_column_index is not None:
        y_val_raw = y_val_raw[:, target_column_index:target_column_index+1]
    
    X_val_raw, y_val_raw = _filter_finite(X_val_raw, y_val_raw)

    X_test_raw = test_data[:, feature_columns]
    y_test_raw = test_data[:, target_columns]
    
    # Apply single-pollutant processing to test data
    if target_column_index is not None:
        y_test_raw = y_test_raw[:, target_column_index:target_column_index+1]
    
    X_test_raw, y_test_raw = _filter_finite(X_test_raw, y_test_raw)

    if log_transform_targets:
        print(
            f"Applying np.log1p transformation to target columns (relative indices): {log_transform_targets}"
        )

        for tgt_idx in log_transform_targets:
            # Handle single-pollutant case: check if the log transform applies to our selected pollutant
            if target_column_index is not None:
                if tgt_idx == target_column_index:
                    # Apply log transform to the single target (now at index 0)
                    y_train_raw[:, 0] = np.log1p(np.maximum(0, y_train_raw[:, 0]))
                    y_val_raw[:, 0] = np.log1p(np.maximum(0, y_val_raw[:, 0]))
                    y_test_raw[:, 0] = np.log1p(np.maximum(0, y_test_raw[:, 0]))
                # Skip if log transform doesn't apply to our selected pollutant
            else:
                # Multi-pollutant case: original logic
                if tgt_idx < 0 or tgt_idx >= len(target_columns):
                    raise IndexError(
                        f"Target index {tgt_idx} is out of bounds for target_columns of length {len(target_columns)}"
                    )

                y_train_raw[:, tgt_idx] = np.log1p(np.maximum(0, y_train_raw[:, tgt_idx]))
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
        }
    )

    if lat_col_name in V2_FIELD_NAMES and lon_col_name in V2_FIELD_NAMES:
        lat_col_idx = V2_FIELD_NAMES.index(lat_col_name)
        lon_col_idx = V2_FIELD_NAMES.index(lon_col_name)

        # For spatial plotting, use the same filtering logic as applied to the processed data
        if target_column_index is not None:
            # Single-pollutant case: filter based on the selected target column
            target_cols_for_filtering = [target_columns[target_column_index]]
        else:
            # Multi-pollutant case: use all target columns
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

    Each output sample contains *lookback* consecutive rows of *features*,
    ending *just before* the corresponding row of *targets* that serves as the
    prediction label.

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
