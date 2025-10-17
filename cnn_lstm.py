#!/usr/bin/env python3

import argparse
from pathlib import Path
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gc
from datetime import datetime
import warnings

import mlflow
import tensorflow as tf

# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

# Try importing shap for feature importance analysis
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn(
        "SHAP package not found. SHAP analysis will be skipped. "
        "Install with 'pip install shap' to enable feature importance analysis."
    )

from src import (
    data_loader,
    train,
    evaluate,
)

from src.utils import set_global_seed


def remove_outliers_by_percentile(
    data: np.ndarray,
    target_column_index: int,
    percentile_threshold: float = 10.0,
    stage: str = "before",
) -> tuple:
    """
    Remove outliers from data based on percentile thresholds.

    Args:
        data (np.ndarray): Input data array
        target_column_index (int): Index of the target column to use for outlier detection
        percentile_threshold (float): Percentile threshold (removes lowest and highest X%)
        stage (str): When outlier removal is applied ("before" or "after" normalization)

    Returns:
        tuple: (filtered_data, outlier_mask, outlier_stats)
    """
    if percentile_threshold <= 0:
        return data, np.ones(len(data), dtype=bool), {}

    target_values = data[:, target_column_index]

    # Calculate percentile thresholds
    lower_threshold = np.percentile(target_values, percentile_threshold)
    upper_threshold = np.percentile(target_values, 100 - percentile_threshold)

    # Create mask for non-outliers
    outlier_mask = (target_values >= lower_threshold) & (
        target_values <= upper_threshold
    )

    # Filter data
    filtered_data = data[outlier_mask]

    # Calculate statistics
    outlier_stats = {
        "original_count": len(data),
        "filtered_count": len(filtered_data),
        "removed_count": len(data) - len(filtered_data),
        "removal_percentage": ((len(data) - len(filtered_data)) / len(data)) * 100,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
        "percentile_threshold": percentile_threshold,
        "stage": stage,
    }

    print(f"Outlier removal ({stage} normalization):")
    print(f"  Original samples: {outlier_stats['original_count']:,}")
    print(f"  Filtered samples: {outlier_stats['filtered_count']:,}")
    print(
        f"  Removed samples: {outlier_stats['removed_count']:,} ({outlier_stats['removal_percentage']:.2f}%)"
    )
    print(f"  Thresholds: [{lower_threshold:.4f}, {upper_threshold:.4f}]")

    return filtered_data, outlier_mask, outlier_stats


def split_data_by_strategy(
    raw_data: np.ndarray,
    split_strategy: str = "chronological",
    test_region: str = None,
    year_column: int = 2,
    train_years: str = None,
    val_years: str = None,
    test_years: str = None,
    train_start: int = 2001,
    train_end: int = 2012,
    val_year: int = 2013,
    test_start: int = 2014,
    test_end: int = 2015,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Split data based on the specified strategy with enhanced configuration options.

    Args:
        raw_data (np.ndarray): The raw data to split
        split_strategy (str): The splitting strategy to use ('chronological', 'random', 'regional', 'rolling_origin')
        test_region (str): Test region bounding box as 'min_lon,min_lat,max_lon,max_lat'
        year_column (int): Index of the year column in the data
        train_years (str): Comma-separated list of training years (for chronological split)
        val_years (str): Comma-separated list of validation years (for chronological split)
        test_years (str): Comma-separated list of test years (for chronological split)
        train_start (int): Start year for training range (for chronological split)
        train_end (int): End year for training range (for chronological split)
        val_year (int): Single validation year (for chronological split)
        test_start (int): Start year for test range (for chronological split)
        test_end (int): End year for test range (for chronological split)
        train_frac (float): Fraction of data for training (for random split)
        val_frac (float): Fraction of data for validation (for random split)
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_data, val_data, test_data, split_config)
    """
    from src.split_strategies import (
        DataSplitter,
        create_chronological_split_config,
        create_random_split_config,
        create_regional_split_config,
        create_rolling_origin_split_config,
    )

    # Parse year lists if provided
    train_years_list = None
    val_years_list = None
    test_years_list = None

    if train_years:
        train_years_list = [int(y.strip()) for y in train_years.split(",")]
    if val_years:
        val_years_list = [int(y.strip()) for y in val_years.split(",")]
    if test_years:
        test_years_list = [int(y.strip()) for y in test_years.split(",")]

    # Create appropriate split configuration
    if test_region:
        # Regional split
        try:
            min_lon, min_lat, max_lon, max_lat = map(float, test_region.split(","))
            test_region_bbox = (min_lon, min_lat, max_lon, max_lat)

            split_config = create_regional_split_config(
                test_region_bbox=test_region_bbox,
                train_val_split_method=split_strategy,
                train_frac=train_frac,
                val_frac=val_frac,
                seed=seed,
            )
        except ValueError as e:
            raise ValueError(f"Invalid test region bounding box: {e}")

    elif split_strategy.lower() == "random":
        # Random split
        split_config = create_random_split_config(
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed,
        )

    elif split_strategy.lower() == "rolling_origin":
        # Rolling origin split
        split_config = create_rolling_origin_split_config(
            year_column_index=year_column,
        )

    else:
        # Chronological split (default)
        split_config = create_chronological_split_config(
            train_years=train_years_list,
            val_years=val_years_list,
            test_years=test_years_list,
            train_start=train_start,
            train_end=train_end,
            val_year=val_year,
            test_start=test_start,
            test_end=test_end,
            year_column_index=year_column,
        )

    # Apply the split
    splitter = DataSplitter(split_config)
    train_data, val_data, test_data = splitter.split(raw_data)

    return train_data, val_data, test_data, splitter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/input_with_geo_and_interactions_v5.npy",
    )
    parser.add_argument(
        "--year-column",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    # Enhanced data splitting arguments
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["chronological", "random", "regional", "rolling_origin"],
        default="chronological",
        help="Data splitting strategy",
    )
    parser.add_argument(
        "--test-region",
        type=str,
        help="Test region bounding box as 'min_lon,min_lat,max_lon,max_lat' (e.g. '-125,32,-115,42')",
    )

    # Chronological split options
    parser.add_argument(
        "--train-years",
        type=str,
        help="Comma-separated list of training years (e.g. '2001,2002,2003')",
    )
    parser.add_argument(
        "--val-years",
        type=str,
        help="Comma-separated list of validation years (e.g. '2013')",
    )
    parser.add_argument(
        "--test-years",
        type=str,
        help="Comma-separated list of test years (e.g. '2014,2015')",
    )
    parser.add_argument(
        "--train-start",
        type=int,
        default=2001,
        help="Start year for training range (default: 2001)",
    )
    parser.add_argument(
        "--train-end",
        type=int,
        default=2012,
        help="End year for training range (default: 2012)",
    )
    parser.add_argument(
        "--val-year",
        type=int,
        default=2013,
        help="Single validation year (default: 2013)",
    )
    parser.add_argument(
        "--test-start",
        type=int,
        default=2014,
        help="Start year for test range (default: 2014)",
    )
    parser.add_argument(
        "--test-end",
        type=int,
        default=2015,
        help="End year for test range (default: 2015)",
    )

    # Random split options
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of data for training (default: 0.7)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--truncate-ozone-percentile",
        type=float,
        default=10.0,
        help="Percentile threshold for ozone outlier truncation (0-100)",
    )

    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP analysis (useful for faster evaluation or if SHAP is not installed)",
    )

    # Individual pollutant selection arguments
    parser.add_argument(
        "--ozone",
        action="store_true",
        help="Train/evaluate only the Ozone model",
    )
    parser.add_argument(
        "--pm25",
        action="store_true",
        help="Train/evaluate only the PM2.5 model",
    )
    parser.add_argument(
        "--no2",
        action="store_true",
        help="Train/evaluate only the NO2 model",
    )

    # Outlier removal arguments
    parser.add_argument(
        "--remove-outliers-percentile",
        type=float,
        default=10.0,
        help="Percentile threshold for outlier removal (removes lowest and highest X%%, default: 10.0)",
    )
    parser.add_argument(
        "--outlier-removal-stage",
        type=str,
        choices=["before", "after"],
        default="before",
        help="When to remove outliers: 'before' normalization or 'after' normalization (default: before)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set reproducibility
    set_global_seed(42)

    # Validate command-line arguments
    if args.truncate_ozone_percentile is not None and not (
        0 <= args.truncate_ozone_percentile <= 100
    ):
        raise ValueError(
            f"truncate-ozone-percentile must be between 0 and 100, got {args.truncate_ozone_percentile}"
        )

    if args.remove_outliers_percentile is not None and not (
        0 <= args.remove_outliers_percentile <= 50
    ):
        raise ValueError(
            f"remove-outliers-percentile must be between 0 and 50, got {args.remove_outliers_percentile}"
        )

    # Validate pollutant selection - only one can be selected at a time
    pollutant_flags = [args.ozone, args.pm25, args.no2]
    if sum(pollutant_flags) > 1:
        raise ValueError(
            "Only one pollutant can be selected at a time. Use --ozone, --pm25, or --no2, not multiple."
        )

    if args.test_region:
        try:
            min_lon, min_lat, max_lon, max_lat = map(float, args.test_region.split(","))
            if min_lon >= max_lon or min_lat >= max_lat:
                raise ValueError(
                    f"Invalid bounding box: min_lon ({min_lon}) must be less than max_lon ({max_lon}) "
                    f"and min_lat ({min_lat}) must be less than max_lat ({max_lat})"
                )
        except ValueError as e:
            if "not enough values to unpack" in str(
                e
            ) or "too many values to unpack" in str(e):
                raise ValueError(
                    f"test-region must be in format 'min_lon,min_lat,max_lon,max_lat', got {args.test_region}"
                )
            raise

    # Data loading
    print("Loading and preprocessing data...")
    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load raw data
    raw_data = np.load(data_path)
    print(f"Raw data shape: {raw_data.shape}")

    # Split data based on selected strategy with enhanced options
    train_data, val_data, test_data, data_splitter = split_data_by_strategy(
        raw_data,
        split_strategy=args.split_strategy,
        test_region=args.test_region,
        year_column=args.year_column,
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years,
        train_start=args.train_start,
        train_end=args.train_end,
        val_year=args.val_year,
        test_start=args.test_start,
        test_end=args.test_end,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    # Define the path to your US states shapefile for spatial maps
    shapefile_for_map = "data/cb/cb_2018_us_state_20m.shp"

    # Load feature and target names from final_column_names.json
    print("Loading feature and target names from final_column_names.json...")
    with open("data/final_column_names.json", "r") as f:
        # Read the column names and convert them to lowercase for case-insensitive matching
        column_names = [col.lower() for col in json.load(f)]

    # Define lowercase target names to search for
    target_name_variants = {
        "ozone": ["ozone", "ozone_concentration"],
        "pm2.5": ["pm2.5", "pm25_concentration"],
        "no2": ["no2", "no2_concentration"],
    }

    # Find the indices of the target columns
    try:
        ozone_idx = next(
            i
            for i, name in enumerate(column_names)
            if name in target_name_variants["ozone"]
        )
        pm25_idx = next(
            i
            for i, name in enumerate(column_names)
            if name in target_name_variants["pm2.5"]
        )
        no2_idx = next(
            i
            for i, name in enumerate(column_names)
            if name in target_name_variants["no2"]
        )
        target_indices = [ozone_idx, pm25_idx, no2_idx]

        # Get the original-case names for logging
        original_column_names = json.load(open("data/final_column_names.json", "r"))
        target_names = [original_column_names[i] for i in target_indices]
        print(
            f"Successfully found target columns: {target_names} at indices {target_indices}"
        )
    except StopIteration:
        raise ValueError(
            "Could not find one or more target columns in final_column_names.json"
        )

    # Define feature names and their indices by EXCLUDING the found targets
    feature_indices = [i for i in range(len(column_names)) if i not in target_indices]
    feature_names = [original_column_names[i] for i in feature_indices]

    print(f"Correctly defined {len(feature_names)} features.")
    print(f"Feature columns: {len(feature_indices)} features")
    print(f"Target columns: {target_indices} (Ozone, PM2.5, NO2)")

    # This `feature_names` list is the one you will pass to your SHAP functions later.

    # Determine which pollutants to process based on command-line arguments
    all_pollutant_configs = {
        "Ozone": {
            "target_index": 0,
            "use_robust_scaler_targets": False,  # StandardScaler for small range
            "log_transform": False,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        "PM2.5": {
            "target_index": 1,
            "use_robust_scaler_targets": True,  # RobustScaler for outliers
            "log_transform": False,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        "NO2": {
            "target_index": 2,
            "use_robust_scaler_targets": True,  # RobustScaler for outliers
            "log_transform": False,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
    }

    # Filter pollutants based on command-line arguments
    pollutant_configs = {}
    if args.ozone:
        pollutant_configs["Ozone"] = all_pollutant_configs["Ozone"]
        print("Training/evaluating only Ozone model")
    elif args.pm25:
        pollutant_configs["PM2.5"] = all_pollutant_configs["PM2.5"]
        print("Training/evaluating only PM2.5 model")
    elif args.no2:
        pollutant_configs["NO2"] = all_pollutant_configs["NO2"]
        print("Training/evaluating only NO2 model")
    else:
        # Default: process all pollutants
        pollutant_configs = all_pollutant_configs
        print("Training/evaluating all pollutants")

    # Create base results directory
    base_results_dir = Path("test_results")
    base_results_dir.mkdir(exist_ok=True)

    # MLflow setup
    mlflow.set_experiment("CNN_LSTM_Per_Pollutant_Training")

    with mlflow.start_run(
        run_name=f"CNN_LSTM_per_pollutant_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        # Log global parameters
        mlflow.log_param("model_type", "cnn_lstm_per_pollutant_no_lookback")
        mlflow.log_param("data_file", str(data_path))
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("eval_only", args.eval_only)
        mlflow.log_param("skip_shap", args.skip_shap)
        mlflow.log_param("remove_outliers_percentile", args.remove_outliers_percentile)
        mlflow.log_param("outlier_removal_stage", args.outlier_removal_stage)
        mlflow.log_param("truncate_ozone_percentile", args.truncate_ozone_percentile)

        # Log split configuration
        split_config_dict = data_splitter.get_config_dict()
        for key, value in split_config_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    mlflow.log_param(f"split_{key}_{sub_key}", sub_value)
            else:
                mlflow.log_param(f"split_{key}", value)

        # Log which pollutants are being processed
        selected_pollutants = list(pollutant_configs.keys())
        mlflow.log_param("selected_pollutants", ",".join(selected_pollutants))
        mlflow.log_param("single_pollutant_mode", len(selected_pollutants) == 1)

        # Storage for all results
        all_pollutant_models = {}
        all_pollutant_histories = {}
        all_pollutant_data = {}
        all_pollutant_metrics = {}

        # Process data and train/load models for each pollutant
        for pollutant_name, config in pollutant_configs.items():
            print(f"\n{'='*50}")
            print(f"Processing {pollutant_name}")
            print(f"{'='*50}")

            # Apply outlier removal before preprocessing if specified
            current_train_data = train_data.copy()
            current_val_data = val_data.copy()
            current_test_data = test_data.copy()
            outlier_stats = {}

            if (
                args.remove_outliers_percentile > 0
                and args.outlier_removal_stage == "before"
            ):
                print(
                    f"Applying outlier removal BEFORE normalization for {pollutant_name}..."
                )

                # Get the actual target column index in the raw data
                target_col_in_raw_data = target_indices[config["target_index"]]

                # Remove outliers from each dataset
                current_train_data, train_mask, train_stats = (
                    remove_outliers_by_percentile(
                        current_train_data,
                        target_col_in_raw_data,
                        args.remove_outliers_percentile,
                        "before",
                    )
                )
                current_val_data, val_mask, val_stats = remove_outliers_by_percentile(
                    current_val_data,
                    target_col_in_raw_data,
                    args.remove_outliers_percentile,
                    "before",
                )
                current_test_data, test_mask, test_stats = (
                    remove_outliers_by_percentile(
                        current_test_data,
                        target_col_in_raw_data,
                        args.remove_outliers_percentile,
                        "before",
                    )
                )

                outlier_stats = {
                    "train": train_stats,
                    "val": val_stats,
                    "test": test_stats,
                }

            # Preprocess data for this specific pollutant
            print(
                f"Preprocessing data for {pollutant_name} with target index {config['target_index']}..."
            )

            # Apply truncation for Ozone if specified (this is separate from outlier removal)
            truncate_percentile = None
            if pollutant_name == "Ozone" and args.truncate_ozone_percentile > 0:
                truncate_percentile = args.truncate_ozone_percentile
                print(f"Applying {truncate_percentile}% truncation for Ozone")

            single_pollutant_data = data_loader.preprocess_data(
                current_train_data,
                current_val_data,
                current_test_data,
                feature_columns=feature_indices,
                target_columns=target_indices,
                target_column_index=config[
                    "target_index"
                ],  # Single pollutant processing
                truncate_target_percentile=truncate_percentile,
                log_transform_targets=None,  # No log transformation
                use_robust_scaler_targets=config["use_robust_scaler_targets"],
            )

            # Apply outlier removal after preprocessing if specified
            if (
                args.remove_outliers_percentile > 0
                and args.outlier_removal_stage == "after"
            ):
                print(
                    f"Applying outlier removal AFTER normalization for {pollutant_name}..."
                )

                # For after-normalization outlier removal, we work with the normalized targets
                # Combine train, val, test data temporarily for consistent outlier detection
                combined_X = np.vstack(
                    [
                        single_pollutant_data["X_train"],
                        single_pollutant_data["X_val"],
                        single_pollutant_data["X_test"],
                    ]
                )
                combined_y = np.hstack(
                    [
                        single_pollutant_data["y_train"].ravel(),
                        single_pollutant_data["y_val"].ravel(),
                        single_pollutant_data["y_test"].ravel(),
                    ]
                )
                combined_y_raw = np.hstack(
                    [
                        single_pollutant_data["y_train_raw"].ravel(),
                        single_pollutant_data["y_val_raw"].ravel(),
                        single_pollutant_data["y_test_raw"].ravel(),
                    ]
                )

                # Create temporary combined data for outlier detection
                temp_data = np.column_stack([combined_X, combined_y])

                # Remove outliers based on normalized target values (last column)
                filtered_temp_data, outlier_mask, after_stats = (
                    remove_outliers_by_percentile(
                        temp_data,
                        temp_data.shape[1] - 1,
                        args.remove_outliers_percentile,
                        "after",
                    )
                )

                # Split back into train/val/test
                train_size = len(single_pollutant_data["X_train"])
                val_size = len(single_pollutant_data["X_val"])

                train_mask = outlier_mask[:train_size]
                val_mask = outlier_mask[train_size : train_size + val_size]
                test_mask = outlier_mask[train_size + val_size :]

                # Apply masks to filter the data
                single_pollutant_data["X_train"] = single_pollutant_data["X_train"][
                    train_mask
                ]
                single_pollutant_data["y_train"] = single_pollutant_data["y_train"][
                    train_mask
                ]
                single_pollutant_data["y_train_raw"] = single_pollutant_data[
                    "y_train_raw"
                ][train_mask]

                single_pollutant_data["X_val"] = single_pollutant_data["X_val"][
                    val_mask
                ]
                single_pollutant_data["y_val"] = single_pollutant_data["y_val"][
                    val_mask
                ]
                single_pollutant_data["y_val_raw"] = single_pollutant_data["y_val_raw"][
                    val_mask
                ]

                single_pollutant_data["X_test"] = single_pollutant_data["X_test"][
                    test_mask
                ]
                single_pollutant_data["y_test"] = single_pollutant_data["y_test"][
                    test_mask
                ]
                single_pollutant_data["y_test_raw"] = single_pollutant_data[
                    "y_test_raw"
                ][test_mask]

                outlier_stats = {"combined": after_stats}

            # Store outlier removal statistics
            single_pollutant_data["outlier_stats"] = outlier_stats

            # Reshape 2D data to proper 3D for improved CNN+LSTM architecture
            print(f"Reshaping data for improved CNN+LSTM architecture...")
            X_train_reshaped = single_pollutant_data["X_train"].reshape(
                single_pollutant_data["X_train"].shape[0],
                1,
                single_pollutant_data["X_train"].shape[1],
            )
            X_val_reshaped = single_pollutant_data["X_val"].reshape(
                single_pollutant_data["X_val"].shape[0],
                1,
                single_pollutant_data["X_val"].shape[1],
            )
            X_test_reshaped = single_pollutant_data["X_test"].reshape(
                single_pollutant_data["X_test"].shape[0],
                1,
                single_pollutant_data["X_test"].shape[1],
            )

            if not args.eval_only:
                # Training mode
                print("\n==================== Training Mode ====================")
                print(f"\n{'='*60}")
                print(f"Training {pollutant_name} CNN+LSTM Model")
                print(f"{'='*60}")

                # Create results directory for this pollutant
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pollutant_results_dir = (
                    base_results_dir
                    / "cnn_lstm_per_pollutant"
                    / pollutant_name
                    / f"run_{timestamp}"
                )
                pollutant_results_dir.mkdir(parents=True, exist_ok=True)
                print(f"Results will be saved to: {pollutant_results_dir}")

                print(
                    f"Training data shape for {pollutant_name}: {single_pollutant_data['X_train'].shape}"
                )
                print(
                    f"Target data shape for {pollutant_name}: {single_pollutant_data['y_train'].shape}"
                )
                print(
                    f"Using {'RobustScaler' if config['use_robust_scaler_targets'] else 'StandardScaler'} for {pollutant_name}"
                )

                print(
                    f"Reshaped data - Train: {X_train_reshaped.shape}, Val: {X_val_reshaped.shape}, Test: {X_test_reshaped.shape}"
                )

                # Option: Create temporal sample weights for domain adaptation (disabled to prevent overfitting)
                # temporal_weights = np.linspace(0.5, 1.5, X_train_reshaped.shape[0])
                # Instead, use uniform weighting to prevent overfitting to recent patterns
                temporal_weights = None
                print("Using uniform sample weighting to prevent overfitting to recent temporal patterns")
                
                # Train CNN+LSTM model
                print(f"Training CNN+LSTM model for {pollutant_name}...")
                model, history = train.train_single_pollutant_cnn_lstm_model(
                    X_train_reshaped,
                    single_pollutant_data[
                        "y_train"
                    ].ravel(),  # Convert to 1D for single-pollutant training
                    X_val_reshaped,
                    single_pollutant_data[
                        "y_val"
                    ].ravel(),  # Convert to 1D for single-pollutant training
                    pollutant_name=pollutant_name,
                    epochs=config["epochs"],
                    batch_size=config["batch_size"],
                    resume=not args.no_resume,
                    use_generator=False,  # Use arrays directly for per-pollutant training
                    sample_weight=temporal_weights,  # Add temporal weighting for domain adaptation
                )

                # Save model
                model_path = pollutant_results_dir / "cnn_lstm_model.keras"
                model.save(model_path)
                print(f"Model saved to {model_path}")

                # Save training history
                if history and hasattr(history, "history") and history.history:
                    # Save training history plot
                    evaluate.training_history_plot(
                        history,
                        save_path=str(pollutant_results_dir / "training_history.png"),
                        show=False,
                        title=f"CNN+LSTM Training History - {pollutant_name}",
                    )

                    # Save history as JSON
                    def _py(val):
                        if isinstance(val, (np.floating, np.integer)):
                            return val.item()
                        return val

                    hist_serializable = {
                        k: [_py(v) for v in vals] for k, vals in history.history.items()
                    }
                    with open(
                        pollutant_results_dir / "training_history.json", "w"
                    ) as f:
                        json.dump(hist_serializable, f, indent=2)

                # Helper function to convert numpy types to JSON-serializable types
                def make_json_serializable(obj):
                    """Recursively convert numpy types to JSON-serializable types."""
                    if isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {
                            key: make_json_serializable(value)
                            for key, value in obj.items()
                        }
                    elif isinstance(obj, (list, tuple)):
                        return [make_json_serializable(item) for item in obj]
                    else:
                        return obj

                # Save configuration
                config_data = {
                    "pollutant_name": pollutant_name,
                    "timestamp": datetime.now().isoformat(),
                    "configuration": config,
                    "data_shapes": {
                        "original_train": str(single_pollutant_data["X_train"].shape),
                        "reshaped_train": str(X_train_reshaped.shape),
                        "target_train": str(single_pollutant_data["y_train"].shape),
                    },
                    "outlier_removal": {
                        "enabled": args.remove_outliers_percentile > 0,
                        "percentile_threshold": float(args.remove_outliers_percentile),
                        "stage": args.outlier_removal_stage,
                        "statistics": make_json_serializable(
                            single_pollutant_data.get("outlier_stats", {})
                        ),
                    },
                    "split_configuration": make_json_serializable(
                        data_splitter.get_config_dict()
                    ),
                    "command_line_args": {
                        "split_strategy": args.split_strategy,
                        "test_region": args.test_region,
                        "train_years": args.train_years,
                        "val_years": args.val_years,
                        "test_years": args.test_years,
                        "train_start": int(args.train_start),
                        "train_end": int(args.train_end),
                        "val_year": int(args.val_year),
                        "test_start": int(args.test_start),
                        "test_end": int(args.test_end),
                        "train_frac": float(args.train_frac),
                        "val_frac": float(args.val_frac),
                        "seed": int(args.seed),
                        "truncate_ozone_percentile": float(
                            args.truncate_ozone_percentile
                        ),
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                    },
                }

                with open(pollutant_results_dir / "config_validation.json", "w") as f:
                    json.dump(config_data, f, indent=2)

                # Save split configuration separately for easy reuse
                data_splitter.save_config(pollutant_results_dir / "split_config.json")

                # Store history
                all_pollutant_histories[pollutant_name] = history

            else:
                # Evaluation-only mode: load existing model
                print(f"\nLoading saved {pollutant_name} CNN+LSTM model...")

                # Find the most recent model directory for this pollutant
                pollutant_dir = (
                    base_results_dir / "cnn_lstm_per_pollutant" / pollutant_name
                )
                if not pollutant_dir.exists():
                    raise FileNotFoundError(
                        f"No saved models found for {pollutant_name} in {pollutant_dir}"
                    )

                run_dirs = [
                    d
                    for d in pollutant_dir.iterdir()
                    if d.is_dir() and d.name.startswith("run_")
                ]
                if not run_dirs:
                    raise FileNotFoundError(
                        f"No run directories found for {pollutant_name}"
                    )

                latest_run_dir = max(run_dirs, key=lambda x: x.name)
                model_path = latest_run_dir / "cnn_lstm_model.keras"

                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")

                print(f"Loading model from {model_path}")
                model = tf.keras.models.load_model(model_path, compile=False)
                pollutant_results_dir = latest_run_dir

            # Store model and data (for both training and eval-only modes)
            all_pollutant_models[pollutant_name] = model
            all_pollutant_data[pollutant_name] = {
                "processed_data": single_pollutant_data,
                "reshaped_data": {
                    "X_train": X_train_reshaped,
                    "X_val": X_val_reshaped,
                    "X_test": X_test_reshaped,
                },
                "results_dir": pollutant_results_dir,
            }

            print(
                f"{'Completed training' if not args.eval_only else 'Loaded model'} for {pollutant_name}"
            )

            # Memory cleanup
            gc.collect()
            tf.keras.backend.clear_session()

        print(f"\n{'='*60}")
        print("EVALUATION PHASE")
        print(f"{'='*60}")

        # Evaluation phase - combine predictions from selected pollutants
        pollutant_names = list(pollutant_configs.keys())

        y_pred_val_combined = []
        y_pred_test_combined = []
        y_val_raw_combined = []
        y_test_raw_combined = []

        for pollutant_name in pollutant_names:
            print(f"\nEvaluating {pollutant_name} model...")

            config = pollutant_configs[pollutant_name]
            model = all_pollutant_models[pollutant_name]

            # Get data
            single_pollutant_data = all_pollutant_data[pollutant_name]["processed_data"]
            reshaped_data = all_pollutant_data[pollutant_name]["reshaped_data"]
            results_dir = all_pollutant_data[pollutant_name]["results_dir"]

            # Get predictions
            print(f"Making predictions for {pollutant_name}...")
            y_pred_val_single = model.predict(reshaped_data["X_val"], verbose=0)
            y_pred_test_single = model.predict(reshaped_data["X_test"], verbose=0)

            # Ensure predictions are 1D for single-pollutant models
            if y_pred_val_single.ndim > 1:
                y_pred_val_single = y_pred_val_single.ravel()
            if y_pred_test_single.ndim > 1:
                y_pred_test_single = y_pred_test_single.ravel()

            # Transform back to original scale
            target_scaler = single_pollutant_data["target_scaler"]
            y_pred_val_orig = target_scaler.inverse_transform(
                y_pred_val_single.reshape(-1, 1)
            ).ravel()
            y_pred_test_orig = target_scaler.inverse_transform(
                y_pred_test_single.reshape(-1, 1)
            ).ravel()

            # Get raw targets (no sequence trimming needed)
            y_val_raw_single = single_pollutant_data["y_val_raw"].ravel()
            y_test_raw_single = single_pollutant_data["y_test_raw"].ravel()

            # Store for combined evaluation
            y_pred_val_combined.append(y_pred_val_orig)
            y_pred_test_combined.append(y_pred_test_orig)
            y_val_raw_combined.append(y_val_raw_single)
            y_test_raw_combined.append(y_test_raw_single)

            # Generate individual pollutant visualizations
            print(f"Generating visualizations for {pollutant_name}...")

            # Density scatter plot
            n_samples_total = len(y_test_raw_single)
            sample_size = min(5000, n_samples_total)
            if sample_size < n_samples_total:
                sample_idx = np.random.choice(
                    n_samples_total, sample_size, replace=False
                )
                y_test_sample = y_test_raw_single[sample_idx]
                y_pred_sample = y_pred_test_orig[sample_idx]
            else:
                y_test_sample = y_test_raw_single
                y_pred_sample = y_pred_test_orig

            evaluate.density_scatter_plot(
                y_test_sample,
                y_pred_sample,
                pollutant_name=pollutant_name,
                save_path=str(
                    results_dir
                    / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_density_scatter.png"
                ),
                show=False,
            )

            # Residuals plot
            evaluate.residuals_plot(
                y_test_sample,
                y_pred_sample,
                pollutant_name=pollutant_name,
                save_path=str(
                    results_dir
                    / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_residuals.png"
                ),
                show=False,
            )

            # Spatial concentration map - attempt to extract coordinates from test data
            # Check if we have test_data_raw in the dictionary
            if "test_data_raw" in single_pollutant_data:
                test_data = single_pollutant_data["test_data_raw"]

                # Find latitude and longitude columns
                lat_col_idx = None
                lon_col_idx = None

                # Try to find lat/lon columns by name in the first few columns
                # Typically lat/lon are in the first few columns of the dataset
                for i in range(min(5, test_data.shape[1])):
                    col_data = test_data[:, i]
                    # Latitude typically ranges from -90 to 90
                    if np.nanmin(col_data) >= -90 and np.nanmax(col_data) <= 90:
                        lat_col_idx = i
                    # Longitude typically ranges from -180 to 180
                    elif np.nanmin(col_data) >= -180 and np.nanmax(col_data) <= 180:
                        lon_col_idx = i

                # If we found both lat and lon columns, generate the spatial concentration map
                if lat_col_idx is not None and lon_col_idx is not None:
                    print(
                        f"Generating spatial concentration map for {pollutant_name}..."
                    )

                    # Extract coordinates and true values
                    lats = test_data[:, lat_col_idx]
                    lons = test_data[:, lon_col_idx]
                    concentrations = y_test_raw_single

                    # Ensure arrays have matching lengths
                    min_length = min(len(lats), len(lons), len(concentrations))
                    print(
                        f"Spatial map data shapes - lons: {lons.shape}, lats: {lats.shape}, concentrations: {concentrations.shape}"
                    )
                    print(f"Using min length: {min_length} for spatial map")

                    lats = lats[:min_length]
                    lons = lons[:min_length]
                    concentrations = concentrations[:min_length]

                    # **FIX:** Create a mask for valid (non-NaN) data across all arrays
                    valid_mask = (
                        np.isfinite(concentrations)
                        & np.isfinite(lons)
                        & np.isfinite(lats)
                    )

                    # Apply the mask to all arrays before plotting
                    lons_filtered = lons[valid_mask]
                    lats_filtered = lats[valid_mask]
                    concentrations_filtered = concentrations[valid_mask]

                    print(
                        f"After NaN filtering: {len(lons_filtered)} valid spatial points"
                    )

                    # Update arrays to use filtered versions
                    lons = lons_filtered
                    lats = lats_filtered
                    concentrations = concentrations_filtered

                    # Generate the spatial concentration map
                    evaluate.spatial_concentration_map(
                        lons=lons,
                        lats=lats,
                        concentrations=concentrations,
                        pollutant_name=pollutant_name,
                        shapefile_path=shapefile_for_map,  # Pass the shapefile path
                        save_path=str(
                            results_dir
                            / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_concentration_map.png"
                        ),
                        show=False,
                        cmap="plasma",  # Sequential colormap as required
                    )
                    print(f"Spatial concentration map saved for {pollutant_name}")

                    # Generate spatial truth/prediction/bias maps
                    print(f"Generating spatial bias analysis for {pollutant_name}...")

                    # Get predictions for the same spatial points and apply the same mask
                    y_pred_spatial_full = y_pred_test_orig[:min_length]
                    y_true_spatial_full = y_test_raw_single[:min_length]

                    # Apply the same valid_mask to predictions
                    y_pred_spatial = y_pred_spatial_full[valid_mask]
                    y_true_spatial = y_true_spatial_full[valid_mask]

                    # **FIX:** Clip predictions at zero to prevent negative concentrations
                    y_pred_spatial = np.maximum(0, y_pred_spatial)

                    # Generate truth/prediction/bias maps
                    bias_maps_save_path = str(
                        results_dir
                        / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_truth_prediction_bias_maps.png"
                    )

                    evaluate.plot_truth_prediction_bias_maps(
                        lons=lons,
                        lats=lats,
                        y_true=y_true_spatial,
                        y_pred=y_pred_spatial,
                        pollutant_name=pollutant_name,
                        shapefile_path=shapefile_for_map,  # Pass the shapefile path
                        save_path=bias_maps_save_path,
                        show=False,
                    )
                    print(f"Spatial bias maps saved for {pollutant_name}")

                    # Generate bias distribution plot
                    bias_dist_save_path = str(
                        results_dir
                        / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_bias_distribution.png"
                    )

                    evaluate.plot_bias_distribution(
                        y_true=y_test_raw_single,
                        y_pred=y_pred_test_orig,
                        pollutant_name=pollutant_name,
                        save_path=bias_dist_save_path,
                        show=False,
                    )
                    print(f"Bias distribution plot saved for {pollutant_name}")
                else:
                    print(
                        f"Could not identify lat/lon columns for {pollutant_name} spatial map"
                    )
            else:
                print(f"Raw test data not available for {pollutant_name} spatial map")

            # Spatial concentration map - check for dedicated lat/lon fields
            if (
                "lats_test" in single_pollutant_data
                and "lons_test" in single_pollutant_data
            ):
                print(f"Generating spatial concentration map for {pollutant_name}...")

                # Extract coordinates and true values
                lats = single_pollutant_data["lats_test"]
                lons = single_pollutant_data["lons_test"]
                concentrations = y_test_raw_single

                # Ensure arrays have matching lengths
                min_length = min(len(lats), len(lons), len(concentrations))
                print(
                    f"Spatial map data shapes - lons: {lons.shape}, lats: {lats.shape}, concentrations: {concentrations.shape}"
                )
                print(f"Using min length: {min_length} for spatial map")

                lats = lats[:min_length]
                lons = lons[:min_length]
                concentrations = concentrations[:min_length]

                # Generate the spatial concentration map
                evaluate.spatial_concentration_map(
                    lons=lons,
                    lats=lats,
                    concentrations=concentrations,
                    pollutant_name=pollutant_name,
                    shapefile_path=shapefile_for_map,  # Pass the shapefile path
                    save_path=str(
                        results_dir
                        / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_concentration_map.png"
                    ),
                    show=False,
                    cmap="plasma",  # Sequential colormap as required
                    point_size=2,  # Point size 2 as requested
                )
                print(f"Spatial concentration map saved for {pollutant_name}")

                # Generate spatial truth/prediction/bias maps
                print(f"Generating spatial bias analysis for {pollutant_name}...")

                # Get predictions for the same spatial points
                y_pred_spatial = y_pred_test_orig[:min_length]
                y_true_spatial = y_test_raw_single[:min_length]

                # Generate truth/prediction/bias maps
                bias_maps_save_path = str(
                    results_dir
                    / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_truth_prediction_bias_maps.png"
                )

                evaluate.plot_truth_prediction_bias_maps(
                    lons=lons,
                    lats=lats,
                    y_true=y_true_spatial,
                    y_pred=y_pred_spatial,
                    pollutant_name=pollutant_name,
                    shapefile_path=shapefile_for_map,  # Pass the shapefile path
                    save_path=bias_maps_save_path,
                    show=False,
                    point_size=2,  # Point size 2 as requested
                )
                print(f"Spatial bias maps saved for {pollutant_name}")

                # Generate bias distribution plot
                bias_dist_save_path = str(
                    results_dir
                    / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_bias_distribution.png"
                )

                evaluate.plot_bias_distribution(
                    y_true=y_test_raw_single,
                    y_pred=y_pred_test_orig,
                    pollutant_name=pollutant_name,
                    save_path=bias_dist_save_path,
                    show=False,
                )
                print(f"Bias distribution plot saved for {pollutant_name}")
            else:
                print(
                    f"Dedicated lat/lon fields not available for {pollutant_name} spatial map"
                )

            # SHAP Analysis Integration
            if not args.skip_shap and SHAP_AVAILABLE:
                try:
                    print(f"Starting SHAP analysis for {pollutant_name}...")

                    # Use small subsets of SCALED data
                    background_data = X_train_reshaped[
                        np.random.choice(X_train_reshaped.shape[0], 100, replace=False)
                    ]
                    test_data_subset = X_test_reshaped[
                        np.random.choice(X_test_reshaped.shape[0], 500, replace=False)
                    ]

                    # 1. Calculate SHAP values ONCE
                    explainer = shap.GradientExplainer(model, background_data)
                    shap_values_4d = explainer.shap_values(test_data_subset)
                    if isinstance(shap_values_4d, list):
                        shap_values_4d = shap_values_4d[0]
                    shap_values_3d = np.squeeze(shap_values_4d)

                    # Handle different SHAP value shapes
                    if shap_values_3d.ndim == 2:
                        # Already 2D (samples, features) - use as is
                        shap_values_2d = shap_values_3d
                    elif shap_values_3d.ndim == 3:
                        if shap_values_3d.shape[1] == 1:
                            # Shape is (samples, 1, features) -> squeeze to (samples, features)
                            shap_values_2d = shap_values_3d.squeeze(axis=1)
                        else:
                            # Multiple timesteps - average over timesteps
                            shap_values_2d = np.mean(shap_values_3d, axis=1)
                    else:
                        raise ValueError(
                            f"Unexpected SHAP values shape: {shap_values_3d.shape}"
                        )

                    # Handle test data subset
                    if test_data_subset.ndim == 3 and test_data_subset.shape[1] == 1:
                        # Shape is (samples, 1, features) -> squeeze to (samples, features)
                        features_2d = test_data_subset.squeeze(axis=1)
                    else:
                        # Multiple timesteps - average over timesteps or already 2D
                        features_2d = (
                            np.mean(test_data_subset, axis=1)
                            if test_data_subset.ndim == 3
                            else test_data_subset
                        )

                    # Verify shapes are correct for SHAP plotting
                    assert (
                        shap_values_2d.ndim == 2
                    ), f"shap_values_2d must be 2D, got shape {shap_values_2d.shape}"
                    assert (
                        features_2d.ndim == 2
                    ), f"features_2d must be 2D, got shape {features_2d.shape}"
                    assert shap_values_2d.shape[1] == len(
                        feature_names
                    ), f"SHAP values features ({shap_values_2d.shape[1]}) must match feature names ({len(feature_names)})"

                    # 3. Call the plotting functions with the correct data
                    # Call the simplified dot plot function
                    shap_dot_save_path = str(
                        results_dir / f"shap_dot_plot_{pollutant_name}.png"
                    )
                    evaluate.shap_summary_plot(
                        shap_values_2d=shap_values_2d,
                        features_2d=features_2d,
                        feature_names=feature_names,  # Correct list from Step 1
                        pollutant_name=pollutant_name,
                        save_path=shap_dot_save_path,
                        show=False,
                    )

                    # Call the bar plot function
                    shap_bar_save_path = str(
                        results_dir / f"shap_global_bar_plot_{pollutant_name}.png"
                    )

                    # For bar plot, we need to handle the shape properly
                    if shap_values_3d.ndim == 2:
                        # If SHAP values are 2D, add a timestep dimension for the bar plot function
                        shap_values_for_bar = shap_values_3d[
                            :, np.newaxis, :
                        ]  # (samples, 1, features)
                    else:
                        shap_values_for_bar = shap_values_3d

                    evaluate.shap_global_importance_bar_plot(
                        shap_values=shap_values_for_bar,
                        feature_names=feature_names,  # Correct list from Step 1
                        pollutant_name=pollutant_name,
                        save_path=shap_bar_save_path,
                        show=False,
                    )

                    print(f"SHAP analysis complete for {pollutant_name}.")

                    # Calculate feature importance from SHAP values for saving
                    if shap_values_3d.ndim == 2:
                        # For 2D SHAP values, average over samples only
                        importance_values = np.abs(shap_values_3d).mean(axis=0)
                    else:
                        # For 3D SHAP values, average over samples and timesteps
                        importance_values = np.abs(shap_values_3d).mean(axis=(0, 1))

                    feature_importance = {
                        feature: importance
                        for feature, importance in zip(feature_names, importance_values)
                    }

                    # Save feature importance as JSON
                    importance_file = (
                        results_dir
                        / f"{pollutant_name.lower()}_feature_importance.json"
                    )

                    # Convert numpy values to Python native types
                    def convert_numpy_value(val):
                        if isinstance(val, np.ndarray):
                            if val.size == 1:
                                return val.item()
                            return val.tolist()
                        elif isinstance(val, (np.floating, np.integer)):
                            return val.item()
                        return val

                    with open(importance_file, "w") as f:
                        json.dump(
                            {
                                "feature_importance": {
                                    feature: convert_numpy_value(importance)
                                    for feature, importance in feature_importance.items()
                                },
                                "model_type": "CNN+LSTM",
                                "pollutant": pollutant_name,
                                "background_sample_size": 100,
                                "timestamp": datetime.now().isoformat(),
                            },
                            f,
                            indent=2,
                        )

                    # Log top 10 most important features to MLflow
                    sorted_importance = sorted(
                        feature_importance.items(),
                        key=lambda x: convert_numpy_value(x[1]),
                        reverse=True,
                    )[:10]

                    for i, (feature, importance) in enumerate(sorted_importance):
                        mlflow.log_metric(
                            f"{pollutant_name.lower()}_top{i+1}_feature",
                            convert_numpy_value(importance),
                        )
                        mlflow.log_param(
                            f"{pollutant_name.lower()}_top{i+1}_feature_name", feature
                        )

                    print(f"SHAP analysis completed and saved for {pollutant_name}")

                except Exception as e:
                    print(
                        f"Error generating SHAP analysis for {pollutant_name}: {str(e)}"
                    )
                    print("Continuing with other evaluations...")
            elif args.skip_shap:
                print(
                    f"Skipping SHAP analysis for {pollutant_name} (--skip-shap flag used)"
                )
            elif not SHAP_AVAILABLE:
                print(
                    f"Skipping SHAP analysis for {pollutant_name} (SHAP package not installed)"
                )

            print(f"Visualizations saved for {pollutant_name}")

        # Combine all predictions for overall evaluation
        print("\nCombining predictions for overall evaluation...")

        # Find minimum length across all pollutants
        min_val_len = min(len(arr) for arr in y_pred_val_combined)
        min_test_len = min(len(arr) for arr in y_pred_test_combined)

        # Trim all arrays to minimum length
        y_pred_val_combined = [arr[:min_val_len] for arr in y_pred_val_combined]
        y_val_raw_combined = [arr[:min_val_len] for arr in y_val_raw_combined]
        y_pred_test_combined = [arr[:min_test_len] for arr in y_pred_test_combined]
        y_test_raw_combined = [arr[:min_test_len] for arr in y_test_raw_combined]

        # Stack to create multi-pollutant arrays
        y_pred_val_orig = np.column_stack(y_pred_val_combined)
        y_pred_test_orig = np.column_stack(y_pred_test_combined)
        y_val_raw = np.column_stack(y_val_raw_combined)
        y_test_raw = np.column_stack(y_test_raw_combined)

        print(
            f"Combined prediction shapes - Val: {y_pred_val_orig.shape}, Test: {y_pred_test_orig.shape}"
        )

        # Calculate comprehensive metrics
        print("\nCalculating comprehensive metrics...")
        val_metrics = evaluate.calculate_summary_metrics(
            y_val_raw, y_pred_val_orig, pollutant_names
        )
        test_metrics = evaluate.calculate_summary_metrics(
            y_test_raw, y_pred_test_orig, pollutant_names
        )

        # Store metrics for each pollutant
        for pollutant in pollutant_names:
            all_pollutant_metrics[pollutant] = {
                "validation": val_metrics[pollutant],
                "test": test_metrics[pollutant],
            }

        # Display results with enhanced formatting
        print("\n" + "=" * 60)
        print("FINAL EVALUATION METRICS")
        print("=" * 60)

        for pollutant in pollutant_names:
            print(f"\n--- {pollutant} ---")
            # Validation Metrics
            print(f"  Validation RMSE: {val_metrics[pollutant]['RMSE']:.4f}")
            print(f"  Validation R²:   {val_metrics[pollutant]['R2']:.4f}")
            print(f"  Validation MAE:  {val_metrics[pollutant]['MAE']:.4f}")
            print(f"  Validation Bias: {val_metrics[pollutant]['Bias']:.4f}")

            # Test Metrics
            print(f"  Test R²:         {test_metrics[pollutant]['R2']:.4f}")
            print(f"  Test RMSE:       {test_metrics[pollutant]['RMSE']:.2f}")
            print(f"  Test MAE:        {test_metrics[pollutant]['MAE']:.2f}")
            print(f"  Test Bias:       {test_metrics[pollutant]['Bias']:.2f}")

            # Normalized Metrics
            print("  ----------- Normalized -----------")
            nrmse_pct = test_metrics[pollutant].get("NRMSE", float("nan")) * 100
            cv_rmse_pct = test_metrics[pollutant].get("CV_RMSE", float("nan")) * 100
            norm_mae_pct = test_metrics[pollutant].get("Norm_MAE", float("nan")) * 100
            norm_bias_pct = test_metrics[pollutant].get("Norm_Bias", float("nan")) * 100

            print(f"  NRMSE (% of Range):   {nrmse_pct:.2f}%")
            print(f"  CV(RMSE) (% of Mean): {cv_rmse_pct:.2f}%")
            print(f"  Norm MAE (% of Mean): {norm_mae_pct:.2f}%")
            print(f"  Norm Bias (% of Mean):{norm_bias_pct:+.2f}%")

        # Save individual pollutant metrics reports
        for pollutant in pollutant_names:
            results_dir = all_pollutant_data[pollutant]["results_dir"]

            # Save metrics report
            metrics_file = results_dir / "metrics_report.json"

            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, (np.floating, np.integer)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            report = {
                "pollutant_name": pollutant,
                "timestamp": datetime.now().isoformat(),
                "validation_metrics": convert_numpy_types(
                    all_pollutant_metrics[pollutant]["validation"]
                ),
                "test_metrics": convert_numpy_types(
                    all_pollutant_metrics[pollutant]["test"]
                ),
                "normalized_metrics_explanation": {
                    "NRMSE": "Normalized RMSE (RMSE / range of true values)",
                    "CV_RMSE": "Coefficient of Variation of RMSE (RMSE / mean of true values)",
                    "Norm_MAE": "Normalized MAE (MAE / mean of true values)",
                    "Norm_Bias": "Normalized Bias (Bias / mean of true values)",
                },
            }

            with open(metrics_file, "w") as f:
                json.dump(report, f, indent=2)

            # Save training summary
            summary_file = results_dir / "training_summary.json"

            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, (np.floating, np.integer)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            summary_data = {
                "pollutant_name": pollutant,
                "model_type": "cnn_lstm_no_lookback",
                "training_completed": True,
                "final_metrics": convert_numpy_types(all_pollutant_metrics[pollutant]),
                "configuration": convert_numpy_types(pollutant_configs[pollutant]),
            }

            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)

        # MLflow logging
        print("\n===== MLflow Logging =====")

        # Log individual pollutant metrics
        for pollutant in pollutant_names:
            pollutant_key = pollutant.lower().replace(".", "").replace(" ", "_")

            # Test metrics
            mlflow.log_metric(
                f"test_rmse_{pollutant_key}", test_metrics[pollutant]["RMSE"]
            )
            mlflow.log_metric(f"test_r2_{pollutant_key}", test_metrics[pollutant]["R2"])
            mlflow.log_metric(
                f"test_mae_{pollutant_key}", test_metrics[pollutant]["MAE"]
            )
            mlflow.log_metric(
                f"test_bias_{pollutant_key}", test_metrics[pollutant]["Bias"]
            )

            # Normalized metrics
            mlflow.log_metric(
                f"test_nrmse_{pollutant_key}",
                test_metrics[pollutant].get("NRMSE", float("nan")),
            )
            mlflow.log_metric(
                f"test_cv_rmse_{pollutant_key}",
                test_metrics[pollutant].get("CV_RMSE", float("nan")),
            )
            mlflow.log_metric(
                f"test_norm_mae_{pollutant_key}",
                test_metrics[pollutant].get("Norm_MAE", float("nan")),
            )
            mlflow.log_metric(
                f"test_norm_bias_{pollutant_key}",
                test_metrics[pollutant].get("Norm_Bias", float("nan")),
            )

            # Log outlier removal statistics if available
            if pollutant in all_pollutant_data:
                outlier_stats = all_pollutant_data[pollutant]["processed_data"].get(
                    "outlier_stats", {}
                )
                if outlier_stats:
                    if args.outlier_removal_stage == "before":
                        # Log statistics for each dataset split
                        for split in ["train", "val", "test"]:
                            if split in outlier_stats:
                                stats = outlier_stats[split]
                                mlflow.log_metric(
                                    f"{pollutant_key}_outliers_removed_{split}",
                                    stats.get("removed_count", 0),
                                )
                                mlflow.log_metric(
                                    f"{pollutant_key}_outliers_removal_pct_{split}",
                                    stats.get("removal_percentage", 0),
                                )
                    else:
                        # Log combined statistics for after-normalization removal
                        if "combined" in outlier_stats:
                            stats = outlier_stats["combined"]
                            mlflow.log_metric(
                                f"{pollutant_key}_outliers_removed_combined",
                                stats.get("removed_count", 0),
                            )
                            mlflow.log_metric(
                                f"{pollutant_key}_outliers_removal_pct_combined",
                                stats.get("removal_percentage", 0),
                            )

            print(f"Logged metrics for {pollutant}")

        # Log aggregate metrics
        avg_test_rmse = np.mean([test_metrics[p]["RMSE"] for p in pollutant_names])
        avg_test_r2 = np.mean([test_metrics[p]["R2"] for p in pollutant_names])
        avg_test_mae = np.mean([test_metrics[p]["MAE"] for p in pollutant_names])
        avg_test_bias = np.mean([test_metrics[p]["Bias"] for p in pollutant_names])

        mlflow.log_metric("avg_test_rmse", avg_test_rmse)
        mlflow.log_metric("avg_test_r2", avg_test_r2)
        mlflow.log_metric("avg_test_mae", avg_test_mae)
        mlflow.log_metric("avg_test_bias", avg_test_bias)

        print("Aggregate metrics logged to MLflow")

        # Save final combined results
        combined_results_dir = base_results_dir / "cnn_lstm_combined_results"
        combined_results_dir.mkdir(exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        final_results = {
            "timestamp": datetime.now().isoformat(),
            "model_type": "cnn_lstm_per_pollutant_no_lookback",
            "training_mode": "full_training" if not args.eval_only else "eval_only",
            "pollutants": pollutant_names,
            "configurations": convert_numpy_types(pollutant_configs),
            "final_metrics": {
                "validation": {
                    p: convert_numpy_types(val_metrics[p]) for p in pollutant_names
                },
                "test": {
                    p: convert_numpy_types(test_metrics[p]) for p in pollutant_names
                },
            },
            "aggregate_metrics": {
                "avg_test_rmse": convert_numpy_types(avg_test_rmse),
                "avg_test_r2": convert_numpy_types(avg_test_r2),
                "avg_test_mae": convert_numpy_types(avg_test_mae),
                "avg_test_bias": convert_numpy_types(avg_test_bias),
            },
            "data_processing": {
                "split_strategy": args.split_strategy,
                "test_region": args.test_region,
                "outlier_removal": {
                    "enabled": args.remove_outliers_percentile > 0,
                    "percentile_threshold": args.remove_outliers_percentile,
                    "stage": args.outlier_removal_stage,
                },
                "truncate_ozone_percentile": args.truncate_ozone_percentile,
            },
            "selected_pollutants": list(pollutant_configs.keys()),
            "single_pollutant_mode": len(pollutant_configs) == 1,
        }

        with open(combined_results_dir / "final_results.json", "w") as f:
            json.dump(final_results, f, indent=2)

        print(f"\nFinal results saved to {combined_results_dir / 'final_results.json'}")

        print("\n" + "=" * 60)
        print("CNN+LSTM PER-POLLUTANT TRAINING COMPLETED")
        print("=" * 60)
        print(f"Results saved in: {base_results_dir / 'cnn_lstm_per_pollutant'}")
        print(f"Combined results: {combined_results_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
