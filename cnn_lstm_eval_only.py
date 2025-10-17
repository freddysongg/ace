#!/usr/bin/env python3

import argparse
from pathlib import Path
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import mlflow
import tensorflow as tf

from src import (
    data_loader,
    evaluate,
)

from src.utils import set_global_seed


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
        "--model-dir",
        type=str,
        default="test_results/cnn_lstm_chrono_regularized",
    )

    # New command-line arguments for enhanced functionality
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["chronological", "random"],
        default="chronological",
        help="Data splitting strategy (chronological or random)",
    )
    parser.add_argument(
        "--test-region",
        type=str,
        help="Test region bounding box as 'min_lon,min_lat,max_lon,max_lat' (e.g. '-125,32,-115,42')",
    )
    parser.add_argument(
        "--truncate-ozone-percentile",
        type=float,
        default=10.0,
        help="Percentile threshold for ozone outlier truncation (0-100)",
    )
    parser.add_argument(
        "--ozone",
        action="store_true",
        help="Evaluate only the ozone model (skip PM2.5 and NO2)",
    )

    return parser.parse_args()


def find_latest_model_dir(base_dir: Path, pollutant_name: str) -> Path:
    """Find the model directory, handling both run-based and direct model structures."""
    pollutant_dir = base_dir
    if not pollutant_dir.exists():
        raise FileNotFoundError(
            f"No saved models found for {pollutant_name} in {pollutant_dir}"
        )

    # Check for run directories first (original structure)
    run_dirs = [
        d for d in pollutant_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]
    if run_dirs:
        latest_run_dir = max(run_dirs, key=lambda x: x.name)
        return latest_run_dir

    # If no run directories, check if model files are directly in base_dir (new structure)
    model_files = list(pollutant_dir.glob("*.keras"))
    if model_files:
        return pollutant_dir

    raise FileNotFoundError(
        f"No model files found for {pollutant_name} in {pollutant_dir}"
    )


def main():
    args = parse_args()

    # Set reproducibility
    set_global_seed(42)

    print("CNN+LSTM Per-Pollutant Evaluation Only")
    print("=" * 50)

    # Validate command-line arguments
    if args.truncate_ozone_percentile is not None and not (
        0 <= args.truncate_ozone_percentile <= 100
    ):
        raise ValueError(
            f"truncate-ozone-percentile must be between 0 and 100, got {args.truncate_ozone_percentile}"
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

    # Split data based on selected strategy
    if args.test_region:
        # Parse test region bounding box
        try:
            min_lon, min_lat, max_lon, max_lat = map(float, args.test_region.split(","))
            test_region_bbox = (min_lon, min_lat, max_lon, max_lat)
            print(f"Using regional split with bounding box: {test_region_bbox}")

            # Use regional split with the specified bounding box
            train_data, val_data, test_data = data_loader.regional_split(
                raw_data,
                test_region_bbox=test_region_bbox,
                train_val_split_method=args.split_strategy,
                year_column_index=args.year_column,
            )
        except ValueError as e:
            raise ValueError(f"Invalid test region bounding box: {e}")
    else:
        # Use the specified split strategy
        if args.split_strategy.lower() == "chronological":
            print(f"Using chronological split strategy")
            train_data, val_data, test_data = data_loader.chronological_split(
                raw_data, year_column_index=args.year_column
            )
        elif args.split_strategy.lower() == "random":
            print(f"Using random split strategy")
            train_data, val_data, test_data = data_loader.random_split(
                raw_data, train_frac=0.8, val_frac=0.1, seed=42
            )
        else:
            raise ValueError(f"Unsupported split strategy: {args.split_strategy}")

    # Load column names to match training script's feature selection
    try:
        with open("data/final_column_names.json", "r") as f:
            column_names = [col.lower() for col in json.load(f)]
    except FileNotFoundError:
        print(
            "Warning: final_column_names.json not found, using fallback feature selection"
        )
        # Fallback to original logic if column names unavailable
        feature_indices = list(range(3, raw_data.shape[1] - 3))
        target_indices = [
            raw_data.shape[1] - 3,
            raw_data.shape[1] - 2,
            raw_data.shape[1] - 1,
        ]
    else:
        # Match training script: find ozone column and exclude it from features
        target_variants = {"ozone": ["ozone", "ozone_concentration"]}
        try:
            ozone_idx = next(
                i
                for i, name in enumerate(column_names)
                if name in target_variants["ozone"]
            )
            print(f"Found ozone column at index {ozone_idx}")

            # Use same feature selection as training script
            all_target_indices = [ozone_idx]  # Single target for compatibility
            feature_indices = [
                i for i in range(len(column_names)) if i not in all_target_indices
            ]

            # FIXED: Use same target column structure as training script
            # Training uses [ozone_idx] for single-pollutant training
            # Evaluation should match this for consistent scaling/normalization
            if args.ozone:
                # For ozone-only evaluation, use same structure as training
                target_indices = all_target_indices  # [ozone_idx]
                print(f"Using ozone-only target indices: {target_indices}")
            else:
                # For multi-pollutant evaluation, use original logic
                target_indices = [
                    raw_data.shape[1] - 3,  # Ozone
                    raw_data.shape[1] - 2,  # PM2.5
                    raw_data.shape[1] - 1,  # NO2
                ]
                print(f"Using multi-pollutant target indices: {target_indices}")

        except StopIteration:
            print(
                "Warning: Could not find ozone column, using fallback feature selection"
            )
            # Fallback to original logic
            feature_indices = list(range(3, raw_data.shape[1] - 3))
            target_indices = [
                raw_data.shape[1] - 3,
                raw_data.shape[1] - 2,
                raw_data.shape[1] - 1,
            ]

    print(f"Feature columns: {len(feature_indices)} features")
    if args.ozone:
        print(f"Target columns: {target_indices} (Ozone only - matches training script)")
    else:
        print(f"Target columns: {target_indices} (Ozone, PM2.5, NO2)")
    print(f"Expected input shape for model: (batch_size, 1, {len(feature_indices)})")
    print(f"Feature selection matches training script: excluding target columns only")

    # Per-pollutant configuration
    # FIXED: Adjust target_index based on evaluation mode
    if args.ozone:
        # For ozone-only evaluation, target_index=0 since target_indices=[ozone_idx]
        pollutant_configs = {
            "Ozone": {
                "target_index": 0,  # First (and only) target in [ozone_idx]
                "use_robust_scaler_targets": False,  # StandardScaler for small range
                "log_transform": False,
            },
        }
    else:
        # For multi-pollutant evaluation, use original indices
        pollutant_configs = {
            "Ozone": {
                "target_index": 0,  # First target in multi-pollutant list
                "use_robust_scaler_targets": False,  # StandardScaler for small range
                "log_transform": False,
            },
            "PM2.5": {
                "target_index": 1,  # Second target in multi-pollutant list
                "use_robust_scaler_targets": True,  # RobustScaler for outliers
                "log_transform": False,
            },
            "NO2": {
                "target_index": 2,  # Third target in multi-pollutant list
                "use_robust_scaler_targets": True,  # RobustScaler for outliers
                "log_transform": False,
            },
        }

    # Model base directory
    model_base_dir = Path(args.model_dir)

    # MLflow setup
    mlflow.set_experiment("CNN_LSTM_Per_Pollutant_Evaluation")

    with mlflow.start_run(
        run_name=f"CNN_LSTM_eval_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        # Log parameters
        mlflow.log_param("model_type", "cnn_lstm_per_pollutant_eval_only")
        mlflow.log_param("data_file", str(data_path))
        mlflow.log_param("model_dir", str(model_base_dir))
        mlflow.log_param("split_strategy", args.split_strategy)

        if args.test_region:
            mlflow.log_param("test_region", args.test_region)

        if args.truncate_ozone_percentile > 0:
            mlflow.log_param(
                "truncate_ozone_percentile", args.truncate_ozone_percentile
            )

        print("\n==================== Loading Models ====================")

        # Storage for all results
        all_pollutant_models = {}
        all_pollutant_data = {}
        all_pollutant_metrics = {}

        # Filter pollutants based on --ozone flag
        if args.ozone:
            pollutants_to_evaluate = ["Ozone"]
            print("\nEvaluating ONLY Ozone model (--ozone flag specified)")
        else:
            pollutants_to_evaluate = list(pollutant_configs.keys())
            print(f"\nEvaluating all pollutants: {pollutants_to_evaluate}")

        # Load existing models for evaluation
        for pollutant_name in pollutants_to_evaluate:
            print(f"\nLoading saved {pollutant_name} CNN+LSTM model...")

            # Find the model directory for this pollutant
            latest_run_dir = find_latest_model_dir(model_base_dir, pollutant_name)

            # Try different model file names
            model_files = [
                "cnn_lstm_model.keras",
                "cnn_lstm_regularized_model.keras",
                "best_model.keras",
            ]

            model_path = None
            for model_file in model_files:
                potential_path = latest_run_dir / model_file
                if potential_path.exists():
                    model_path = potential_path
                    break

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            print(f"Loading model from {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            all_pollutant_models[pollutant_name] = model

            # Preprocess data for this specific pollutant
            print(f"Preprocessing data for {pollutant_name}...")
            config = pollutant_configs[pollutant_name]

            # Apply truncation for Ozone if specified
            truncate_percentile = None
            if pollutant_name == "Ozone" and args.truncate_ozone_percentile > 0:
                truncate_percentile = args.truncate_ozone_percentile
                print(f"Applying {truncate_percentile}% truncation for Ozone")

            single_pollutant_data = data_loader.preprocess_data(
                train_data,
                val_data,
                test_data,
                feature_columns=feature_indices,
                target_columns=target_indices,
                target_column_index=config["target_index"],
                truncate_target_percentile=truncate_percentile,
                log_transform_targets=None,
                use_robust_scaler_targets=config["use_robust_scaler_targets"],
            )

            # Reshape data for CNN+LSTM (single timestep)
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

            all_pollutant_data[pollutant_name] = {
                "processed_data": single_pollutant_data,
                "reshaped_data": {"X_val": X_val_reshaped, "X_test": X_test_reshaped},
                "results_dir": latest_run_dir,
            }

            print(f"✓ {pollutant_name} model and data loaded successfully")

        print(f"\n{'='*60}")
        print("EVALUATION PHASE")
        print(f"{'='*60}")

        # Evaluation phase - combine predictions from evaluated pollutants
        pollutant_names = pollutants_to_evaluate

        y_pred_val_combined = []
        y_pred_test_combined = []
        y_val_raw_combined = []
        y_test_raw_combined = []

        for pollutant_name in pollutant_names:
            print(f"\nEvaluating {pollutant_name} model...")

            model = all_pollutant_models[pollutant_name]
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

            # Create evaluation subdirectory
            eval_dir = results_dir / "evaluation"
            eval_dir.mkdir(exist_ok=True)

            # Try to get spatial coordinates for advanced plots
            try:
                # Load raw data for spatial analysis
                raw_data = np.load(args.data)
                with open("data/final_column_names.json", "r") as f:
                    column_names = [col.lower() for col in json.load(f)]

                # Find coordinate columns
                lon_idx = next(
                    i for i, name in enumerate(column_names) if name.lower() == "lon"
                )
                lat_idx = next(
                    i for i, name in enumerate(column_names) if name.lower() == "lat"
                )

                # Get test data coordinates
                if args.split_strategy.lower() == "chronological":
                    years = raw_data[:, args.year_column]
                    test_mask = (years >= 2014) & (years <= 2015)
                    test_coords = raw_data[test_mask]

                    # Extract coordinates for spatial plots
                    test_lons = test_coords[:, lon_idx]
                    test_lats = test_coords[:, lat_idx]

                    # Ensure we have matching lengths with predictions
                    min_len = min(
                        len(test_lons), len(test_lats), len(y_test_raw_single)
                    )
                    test_lons = test_lons[:min_len]
                    test_lats = test_lats[:min_len]
                    y_test_spatial = y_test_raw_single[:min_len]
                    y_pred_spatial = y_pred_test_orig[:min_len]

                    print(f"Found spatial coordinates: {len(test_lons)} points")
                    has_spatial_data = True
                else:
                    has_spatial_data = False
                    print("Spatial plots only available for chronological splits")

            except Exception as e:
                print(f"Could not load spatial coordinates: {e}")
                has_spatial_data = False

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
                    eval_dir
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
                    eval_dir
                    / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_residuals.png"
                ),
                show=False,
            )

            # Spatial concentration map
            # Check if we have latitude and longitude data available
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

                # **FIX:** Create a mask for valid (non-NaN) data across all arrays
                valid_mask = (
                    np.isfinite(concentrations) & np.isfinite(lons) & np.isfinite(lats)
                )

                # Apply the mask to all arrays before plotting
                lons_filtered = lons[valid_mask]
                lats_filtered = lats[valid_mask]
                concentrations_filtered = concentrations[valid_mask]

                print(f"After NaN filtering: {len(lons_filtered)} valid spatial points")

                # Update arrays to use filtered versions
                lons = lons_filtered
                lats = lats_filtered
                concentrations = concentrations_filtered

                # Generate the spatial concentration map
                evaluate.spatial_concentration_map(
                    lons=lons,
                    lats=lats,
                    shapefile_path="data/cb/cb_2018_us_state_20m.shp",
                    concentrations=concentrations,
                    pollutant_name=pollutant_name,
                    save_path=str(
                        eval_dir
                        / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_concentration_map.png"
                    ),
                    show=False,
                    cmap="plasma",  # Sequential colormap as required
                )
                print(f"Spatial concentration map saved for {pollutant_name}")
            # Try alternative approach if test_data_raw is available
            elif "test_data_raw" in single_pollutant_data:
                test_data = single_pollutant_data["test_data_raw"]

                # Find latitude and longitude columns
                lat_col_idx = None
                lon_col_idx = None

                # Try to find lat/lon columns in the first few columns
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
                        f"Generating spatial concentration map for {pollutant_name} using raw data..."
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
                        shapefile_path="data/cb/cb_2018_us_state_20m.shp",
                        concentrations=concentrations,
                        pollutant_name=pollutant_name,
                        save_path=str(
                            eval_dir
                            / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_concentration_map.png"
                        ),
                        show=False,
                        cmap="plasma",  # Sequential colormap as required
                    )
                    print(f"Spatial concentration map saved for {pollutant_name}")
                else:
                    print(
                        f"Could not identify lat/lon columns for {pollutant_name} spatial map"
                    )
            else:
                print(
                    f"Latitude/longitude data not available for {pollutant_name} spatial map"
                )

            # Generate additional evaluation plots using existing functions from src/evaluate.py
            print(f"Generating additional evaluation plots for {pollutant_name}...")

            # Bias distribution plot
            try:
                evaluate.plot_bias_distribution(
                    y_test_raw_single,
                    y_pred_test_orig,
                    pollutant_name=pollutant_name,
                    save_path=str(
                        eval_dir
                        / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_bias_distribution.png"
                    ),
                    show=False,
                )
                print(f"✓ Bias distribution plot saved for {pollutant_name}")
            except Exception as e:
                print(
                    f"Could not generate bias distribution plot for {pollutant_name}: {e}"
                )

            # Truth-Prediction bias maps (if spatial data available)
            if has_spatial_data and "test_lons" in locals() and "test_lats" in locals():
                try:
                    evaluate.plot_truth_prediction_bias_maps(
                        test_lons,
                        test_lats,
                        y_test_spatial,
                        y_pred_spatial,
                        pollutant_name=pollutant_name,
                        shapefile_path="data/cb/cb_2018_us_state_20m.shp",
                        save_path=str(
                            eval_dir
                            / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_truth_prediction_bias_maps.png"
                        ),
                        show=False,
                    )
                    print(f"✓ Truth-prediction bias maps saved for {pollutant_name}")
                except Exception as e:
                    print(
                        f"Could not generate truth-prediction bias maps for {pollutant_name}: {e}"
                    )

            # SHAP analysis (if shap is available)
            try:
                import shap

                print(f"Generating SHAP analysis for {pollutant_name}...")

                # Sample data for SHAP analysis (limit for computational efficiency)
                sample_size = min(1000, reshaped_data["X_test"].shape[0])
                sample_indices = np.random.choice(
                    reshaped_data["X_test"].shape[0], sample_size, replace=False
                )
                X_shap = reshaped_data["X_test"][sample_indices]

                # Create SHAP explainer
                explainer = shap.KernelExplainer(
                    model.predict, X_shap[:100]
                )  # Use subset as background
                shap_values = explainer.shap_values(
                    X_shap[:200]
                )  # Analyze subset for speed

                # Ensure shap_values is correct format (2D for summary plot)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # Take first output if multi-output

                # Flatten if needed for CNN+LSTM (samples, timesteps, features) -> (samples, features)
                if shap_values.ndim == 3:
                    shap_values_2d = shap_values.reshape(shap_values.shape[0], -1)
                    X_shap_2d = X_shap.reshape(X_shap.shape[0], -1)
                else:
                    shap_values_2d = shap_values
                    X_shap_2d = X_shap.reshape(X_shap.shape[0], -1)

                # Generate feature names
                feature_names = [f"feature_{i}" for i in range(X_shap_2d.shape[1])]

                # SHAP summary (dot) plot
                evaluate.shap_summary_plot(
                    shap_values_2d[:200],
                    X_shap_2d[:200],
                    feature_names=feature_names,
                    pollutant_name=pollutant_name,
                    max_display=20,
                    save_path=str(
                        eval_dir
                        / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_shap_dot_plot.png"
                    ),
                    show=False,
                )

                # SHAP global bar plot (if original shap_values was 3D)
                if shap_values.ndim == 3:
                    evaluate.shap_global_importance_bar_plot(
                        shap_values[:200],
                        feature_names=feature_names[
                            : shap_values.shape[2]
                        ],  # Only original feature count
                        pollutant_name=pollutant_name,
                        max_display=20,
                        save_path=str(
                            eval_dir
                            / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_shap_global_bar_plot.png"
                        ),
                        show=False,
                    )

                print(f"✓ SHAP analysis plots saved for {pollutant_name}")

            except ImportError:
                print(
                    f"SHAP not available - skipping SHAP analysis for {pollutant_name}"
                )
            except Exception as e:
                print(f"Could not generate SHAP analysis for {pollutant_name}: {e}")

            print(f"✓ All visualizations saved for {pollutant_name}")

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

        # Save evaluation results
        print("\n===== Saving Evaluation Results =====")

        # Create evaluation results directory
        eval_results_dir = Path("test_results/cnn_lstm_evaluation_results")
        eval_results_dir.mkdir(parents=True, exist_ok=True)

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

        # Save comprehensive evaluation report
        evaluation_report = {
            "timestamp": datetime.now().isoformat(),
            "model_type": "cnn_lstm_per_pollutant_evaluation",
            "data_file": str(data_path),
            "model_directories": {
                pollutant: str(all_pollutant_data[pollutant]["results_dir"])
                for pollutant in pollutant_names
            },
            "pollutant_metrics": {
                pollutant: {
                    "validation": convert_numpy_types(val_metrics[pollutant]),
                    "test": convert_numpy_types(test_metrics[pollutant]),
                }
                for pollutant in pollutant_names
            },
            "aggregate_metrics": {
                "avg_val_rmse": float(
                    np.mean([val_metrics[p]["RMSE"] for p in pollutant_names])
                ),
                "avg_val_r2": float(
                    np.mean([val_metrics[p]["R2"] for p in pollutant_names])
                ),
                "avg_test_rmse": float(
                    np.mean([test_metrics[p]["RMSE"] for p in pollutant_names])
                ),
                "avg_test_r2": float(
                    np.mean([test_metrics[p]["R2"] for p in pollutant_names])
                ),
            },
            "normalized_metrics_explanation": {
                "NRMSE": "Normalized RMSE (RMSE / range of true values)",
                "CV_RMSE": "Coefficient of Variation of RMSE (RMSE / mean of true values)",
                "Norm_MAE": "Normalized MAE (MAE / mean of true values)",
                "Norm_Bias": "Normalized Bias (Bias / mean of true values)",
            },
        }

        with open(eval_results_dir / "evaluation_report.json", "w") as f:
            json.dump(evaluation_report, f, indent=2)

        print(
            f"Evaluation report saved to {eval_results_dir / 'evaluation_report.json'}"
        )

        # MLflow logging
        print("\n===== MLflow Logging =====")

        # Log individual pollutant metrics
        for pollutant in pollutant_names:
            pollutant_key = pollutant.lower().replace(".", "").replace(" ", "_")

            # Validation metrics
            mlflow.log_metric(
                f"val_rmse_{pollutant_key}", val_metrics[pollutant]["RMSE"]
            )
            mlflow.log_metric(f"val_r2_{pollutant_key}", val_metrics[pollutant]["R2"])
            mlflow.log_metric(f"val_mae_{pollutant_key}", val_metrics[pollutant]["MAE"])
            mlflow.log_metric(
                f"val_bias_{pollutant_key}", val_metrics[pollutant]["Bias"]
            )

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

            print(f"✓ Logged metrics for {pollutant}")

        # Log aggregate metrics
        avg_val_rmse = np.mean([val_metrics[p]["RMSE"] for p in pollutant_names])
        avg_val_r2 = np.mean([val_metrics[p]["R2"] for p in pollutant_names])
        avg_test_rmse = np.mean([test_metrics[p]["RMSE"] for p in pollutant_names])
        avg_test_r2 = np.mean([test_metrics[p]["R2"] for p in pollutant_names])

        mlflow.log_metric("avg_val_rmse", avg_val_rmse)
        mlflow.log_metric("avg_val_r2", avg_val_r2)
        mlflow.log_metric("avg_test_rmse", avg_test_rmse)
        mlflow.log_metric("avg_test_r2", avg_test_r2)

        print("✓ Aggregate metrics logged to MLflow")

        # Generate combined visualizations
        print("\nGenerating combined visualizations...")
        combined_results_dir = eval_results_dir / "combined_visualizations"
        combined_results_dir.mkdir(exist_ok=True)

        # Multi-pollutant density scatter plots
        n_samples_total = y_test_raw.shape[0]
        sample_size = min(5000, n_samples_total)
        if sample_size < n_samples_total:
            sample_idx = np.random.choice(n_samples_total, sample_size, replace=False)
            y_test_sample = y_test_raw[sample_idx]
            y_pred_sample = y_pred_test_orig[sample_idx]
        else:
            y_test_sample = y_test_raw
            y_pred_sample = y_pred_test_orig

        evaluate.density_scatter_plots_multi(
            y_test_sample,
            y_pred_sample,
            pollutant_names=pollutant_names,
            save_dir=str(combined_results_dir / "density_scatter"),
            show=False,
        )

        # Multi-pollutant error histograms
        evaluate.prediction_error_histograms_multi(
            y_test_raw,
            y_pred_test_orig,
            pollutant_names=pollutant_names,
            save_dir=str(combined_results_dir / "error_histograms"),
            show=False,
        )

        print(f"✓ Combined visualizations saved to {combined_results_dir}")

        print("\n" + "=" * 60)
        print("CNN+LSTM PER-POLLUTANT EVALUATION COMPLETED")
        print("=" * 60)
        print(f"Evaluation results saved in: {eval_results_dir}")
        print(f"Individual model results: {model_base_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
