#!/usr/bin/env python3
"""
CNN+LSTM model with anti-overfitting improvements.
This script addresses the overfitting issues identified in the training history.
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime
import warnings

# Try importing modules
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available. Logging will be disabled.")

from src import data_loader, evaluate
from src.train import train_single_pollutant_cnn_lstm_model
from src.models import get_cnn_lstm_model_no_pooling_regularized
from src.utils import set_global_seed


def remove_outliers_by_percentile(data, target_column_index, percentile_threshold=10.0):
    """Remove outliers from data based on percentile thresholds."""
    if percentile_threshold <= 0:
        return data, np.ones(len(data), dtype=bool), {}

    target_values = data[:, target_column_index]
    lower_threshold = np.percentile(target_values, percentile_threshold)
    upper_threshold = np.percentile(target_values, 100 - percentile_threshold)
    outlier_mask = (target_values >= lower_threshold) & (
        target_values <= upper_threshold
    )
    filtered_data = data[outlier_mask]

    outlier_stats = {
        "original_count": len(data),
        "filtered_count": len(filtered_data),
        "removed_count": len(data) - len(filtered_data),
        "removal_percentage": ((len(data) - len(filtered_data)) / len(data)) * 100,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
    }

    print(f"Outlier removal:")
    print(f"  Original samples: {outlier_stats['original_count']:,}")
    print(f"  Filtered samples: {outlier_stats['filtered_count']:,}")
    print(
        f"  Removed samples: {outlier_stats['removed_count']:,} ({outlier_stats['removal_percentage']:.2f}%)"
    )

    return filtered_data, outlier_mask, outlier_stats


def split_data_chronological(
    raw_data,
    year_column=2,
    train_start=2001,
    train_end=2012,
    val_year=2013,
    test_start=2014,
    test_end=2015,
):
    """Split data chronologically for temporal evaluation."""
    years = raw_data[:, year_column]

    # Create masks for each split
    train_mask = (years >= train_start) & (years <= train_end)
    val_mask = years == val_year
    test_mask = (years >= test_start) & (years <= test_end)

    train_data = raw_data[train_mask]
    val_data = raw_data[val_mask]
    test_data = raw_data[test_mask]

    print(f"Data split:")
    print(f"  Training: {len(train_data):,} samples ({train_start}-{train_end})")
    print(f"  Validation: {len(val_data):,} samples ({val_year})")
    print(f"  Test: {len(test_data):,} samples ({test_start}-{test_end})")

    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="data/input_with_geo_and_interactions_v5.npy"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--outlier-threshold", type=float, default=10.0)
    parser.add_argument(
        "--output-dir", type=str, default="test_results/cnn_lstm_chrono_regularized"
    )
    parser.add_argument(
        "--ozone",
        action="store_true",
        help="Focus on ozone model training and evaluation only",
    )
    args = parser.parse_args()

    # Set reproducibility
    set_global_seed(42)

    print("=== CNN+LSTM Anti-Overfitting Training ===")
    print("Addressing overfitting issues with improved regularization")
    if args.ozone:
        print("Focus: Ozone model training and evaluation only")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Load data
    print("Loading data...")
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw_data = np.load(data_path)
    print(f"Raw data shape: {raw_data.shape}")

    # Load column names
    with open("data/final_column_names.json", "r") as f:
        column_names = [col.lower() for col in json.load(f)]

    # Find target columns
    target_variants = {"ozone": ["ozone", "ozone_concentration"]}
    try:
        ozone_idx = next(
            i for i, name in enumerate(column_names) if name in target_variants["ozone"]
        )
        print(f"Found ozone column at index {ozone_idx}")
    except StopIteration:
        raise ValueError("Could not find ozone concentration column")

    # Split data chronologically
    train_data, val_data, test_data = split_data_chronological(raw_data)

    # Remove outliers
    if args.outlier_threshold > 0:
        print(f"\Removing outliers ({args.outlier_threshold}% threshold)...")
        train_data, _, train_stats = remove_outliers_by_percentile(
            train_data, ozone_idx, args.outlier_threshold
        )
        val_data, _, val_stats = remove_outliers_by_percentile(
            val_data, ozone_idx, args.outlier_threshold
        )
        test_data, _, test_stats = remove_outliers_by_percentile(
            test_data, ozone_idx, args.outlier_threshold
        )

    # Define feature indices (exclude targets)
    all_target_indices = [ozone_idx]  # Add more if needed
    feature_indices = [
        i for i in range(len(column_names)) if i not in all_target_indices
    ]

    # Preprocess data
    print("Preprocessing data for single-pollutant training...")
    single_pollutant_data = data_loader.preprocess_data(
        train_data,
        val_data,
        test_data,
        feature_columns=feature_indices,
        target_columns=all_target_indices,
        target_column_index=0,  # Single pollutant
        use_robust_scaler_targets=False,  # StandardScaler for ozone
    )

    # Reshape for CNN+LSTM (add sequence dimension)
    print("Reshaping data for CNN+LSTM architecture...")
    X_train = single_pollutant_data["X_train"].reshape(
        single_pollutant_data["X_train"].shape[0],
        1,
        single_pollutant_data["X_train"].shape[1],
    )
    X_val = single_pollutant_data["X_val"].reshape(
        single_pollutant_data["X_val"].shape[0],
        1,
        single_pollutant_data["X_val"].shape[1],
    )
    X_test = single_pollutant_data["X_test"].reshape(
        single_pollutant_data["X_test"].shape[0],
        1,
        single_pollutant_data["X_test"].shape[1],
    )

    y_train = single_pollutant_data["y_train"].ravel()
    y_val = single_pollutant_data["y_val"].ravel()
    y_test = single_pollutant_data["y_test"].ravel()

    print(f"Training data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")

    # Train model with regularized architecture
    print("=" * 60)
    print("TRAINING REGULARIZED CNN+LSTM MODEL")
    print("=" * 60)

    # Create model builder function for regularized architecture
    def regularized_model_builder(input_shape, num_outputs):
        return get_cnn_lstm_model_no_pooling_regularized(input_shape, num_outputs)

    # Train the model
    model, history = train_single_pollutant_cnn_lstm_model(
        X_train,
        y_train,
        X_val,
        y_val,
        pollutant_name="Ozone",
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=False,  # Start fresh to test anti-overfitting
        model_builder=regularized_model_builder,
    )

    # Save model
    model_path = output_dir / "cnn_lstm_regularized_model.keras"
    model.save(model_path)
    print(f"\\nModel saved to: {model_path}")

    # Save training history
    if history and hasattr(history, "history") and history.history:
        # Training history plot
        evaluate.training_history_plot(
            history,
            save_path=str(output_dir / "training_history.png"),
            show=False,
            title="CNN+LSTM Regularized Training History - Ozone",
        )

        # Save history as JSON
        history_data = {
            k: [float(v) for v in vals] for k, vals in history.history.items()
        }
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history_data, f, indent=2)

        print(f"Training history saved to: {output_dir}")

        # Analyze overfitting
        val_losses = history.history["val_loss"]
        train_losses = history.history["loss"]

        if len(val_losses) >= 3:
            # Check if validation loss starts increasing while training loss decreases
            val_trend = val_losses[-1] - val_losses[len(val_losses) // 2]
            train_trend = train_losses[-1] - train_losses[len(train_losses) // 2]

            print(f"\\n=== OVERFITTING ANALYSIS ===")
            print(f"Training loss trend (mid to end): {train_trend:+.6f}")
            print(f"Validation loss trend (mid to end): {val_trend:+.6f}")

            if val_trend > 0 and train_trend < 0:
                print(
                    "⚠️  Signs of overfitting detected: val loss increasing while train loss decreasing"
                )
            elif val_trend > 0.001:
                print("⚠️  Validation loss increasing - potential overfitting")
            else:
                print("✅ Good training behavior - no clear overfitting signs")

            final_gap = val_losses[-1] - train_losses[-1]
            print(f"Final train-val gap: {final_gap:.6f}")

            if final_gap > 0.01:
                print("⚠️  Large train-validation gap suggests overfitting")
            else:
                print("✅ Reasonable train-validation gap")

    # Generate predictions and evaluate
    print("\\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Make predictions
    y_pred_test = model.predict(X_test, verbose=0)
    if y_pred_test.ndim > 1:
        y_pred_test = y_pred_test.ravel()

    # Transform back to original scale
    target_scaler = single_pollutant_data["target_scaler"]
    y_pred_test_orig = target_scaler.inverse_transform(
        y_pred_test.reshape(-1, 1)
    ).ravel()
    y_test_orig = single_pollutant_data["y_test_raw"].ravel()

    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
    test_r2 = r2_score(y_test_orig, y_pred_test_orig)
    test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig)
    test_bias = np.mean(y_pred_test_orig - y_test_orig)

    print(f"\\nTest Metrics:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  Bias: {test_bias:+.4f}")

    # Save evaluation plots
    print("\\nGenerating evaluation plots...")

    # Use subset for plotting if dataset is large
    n_samples = len(y_test_orig)
    if n_samples > 5000:
        plot_indices = np.random.choice(n_samples, 5000, replace=False)
        y_test_plot = y_test_orig[plot_indices]
        y_pred_plot = y_pred_test_orig[plot_indices]
    else:
        y_test_plot = y_test_orig
        y_pred_plot = y_pred_test_orig

    # Density scatter plot
    evaluate.density_scatter_plot(
        y_test_plot,
        y_pred_plot,
        pollutant_name="Ozone",
        save_path=str(output_dir / "ozone_density_scatter.png"),
        show=False,
    )

    # Residuals plot
    evaluate.residuals_plot(
        y_test_plot,
        y_pred_plot,
        pollutant_name="Ozone",
        save_path=str(output_dir / "ozone_residuals.png"),
        show=False,
    )

    # Monthly time series (if possible)
    try:
        evaluate.generate_monthly_time_series_from_raw_data(
            output_dir=str(output_dir),
            pollutant_name="Ozone",
            show=False,
        )
    except Exception as e:
        print(f"Could not generate monthly time series: {e}")

    # Generate additional evaluation plots using existing functions from src/evaluate.py
    print("\nGenerating comprehensive evaluation plots...")

    # Bias distribution plot
    try:
        evaluate.plot_bias_distribution(
            y_test_orig,
            y_pred_test_orig,
            pollutant_name="Ozone",
            save_path=str(output_dir / "ozone_bias_distribution.png"),
            show=False,
        )
        print("✓ Bias distribution plot saved")
    except Exception as e:
        print(f"Could not generate bias distribution plot: {e}")

    # Spatial concentration map and bias maps (if spatial data available)
    try:
        # Load raw data for spatial analysis
        raw_data_for_spatial = np.load(args.data)
        with open("data/final_column_names.json", "r") as f:
            column_names_spatial = [col.lower() for col in json.load(f)]

        # Find coordinate columns
        lon_idx = next(
            i for i, name in enumerate(column_names_spatial) if name.lower() == "lon"
        )
        lat_idx = next(
            i for i, name in enumerate(column_names_spatial) if name.lower() == "lat"
        )

        # Get test data coordinates (2014-2015)
        years = raw_data_for_spatial[:, 2]  # Assuming year column is index 2
        test_mask_spatial = (years >= 2014) & (years <= 2015)
        test_coords_spatial = raw_data_for_spatial[test_mask_spatial]

        # Extract coordinates and ozone concentrations
        test_lons_spatial = test_coords_spatial[:, lon_idx]
        test_lats_spatial = test_coords_spatial[:, lat_idx]
        test_ozone_concentrations = test_coords_spatial[:, ozone_idx]

        # Ensure we have matching lengths
        min_len_spatial = min(
            len(test_lons_spatial), len(test_lats_spatial), len(y_test_orig)
        )
        test_lons_spatial = test_lons_spatial[:min_len_spatial]
        test_lats_spatial = test_lats_spatial[:min_len_spatial]
        test_ozone_concentrations = test_ozone_concentrations[:min_len_spatial]
        y_test_spatial = y_test_orig[:min_len_spatial]
        y_pred_spatial = y_pred_test_orig[:min_len_spatial]

        # Define shapefile path for California boundaries
        shapefile_path = "data/cb/cb_2018_us_state_20m.shp"

        # Generate ozone concentration map with California boundaries
        evaluate.spatial_concentration_map(
            test_lons_spatial,
            test_lats_spatial,
            "data/cb/cb_2018_us_state_20m.shp",
            test_ozone_concentrations,
            pollutant_name="Ozone",
            save_path=str(output_dir / "ozone_concentration_map.png"),
            show=False,
            figsize=(12, 10),
            cmap="plasma",
        )
        print("✓ Ozone concentration map saved")

        # Use the existing function for spatial bias maps with California boundaries
        evaluate.plot_truth_prediction_bias_maps(
            test_lons_spatial,
            test_lats_spatial,
            y_test_spatial,
            y_pred_spatial,
            pollutant_name="Ozone",
            shapefile_path="data/cb/cb_2018_us_state_20m.shp",
            save_path=str(output_dir / "ozone_truth_prediction_bias_maps.png"),
            show=False,
        )
        print("✓ Truth-prediction bias maps saved")

    except Exception as e:
        print(f"Could not generate spatial plots: {e}")

    # SHAP analysis (if shap is available)
    try:
        import shap

        print("Generating SHAP analysis...")

        # Sample data for SHAP analysis (limit for computational efficiency)
        sample_size = min(1000, X_test.shape[0])
        sample_indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
        X_shap = X_test[sample_indices]

        # Create SHAP explainer
        explainer = shap.KernelExplainer(
            model.predict, X_shap[:100]
        )  # Use subset as background
        shap_values = explainer.shap_values(X_shap[:200])  # Analyze subset for speed

        # Ensure shap_values is correct format
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
            shap_values_2d,
            X_shap_2d,
            feature_names=feature_names,
            pollutant_name="Ozone",
            max_display=20,
            save_path=str(output_dir / "ozone_shap_dot_plot.png"),
            show=False,
        )

        # SHAP global bar plot (if original shap_values was 3D)
        if shap_values.ndim == 3:
            evaluate.shap_global_importance_bar_plot(
                shap_values,
                feature_names=feature_names[
                    : shap_values.shape[2]
                ],  # Only original feature count
                pollutant_name="Ozone",
                max_display=20,
                save_path=str(output_dir / "ozone_shap_global_bar_plot.png"),
                show=False,
            )

        print("✓ SHAP analysis plots saved")

    except ImportError:
        print("SHAP not available - skipping SHAP analysis")
    except Exception as e:
        print(f"Could not generate SHAP analysis: {e}")

    # Save final results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "cnn_lstm_regularized",
        "anti_overfitting_techniques": {
            "reduced_model_complexity": "32→64 conv, 32 LSTM, 64→32 dense",
            "increased_dropout": "0.4-0.5 in dense layers",
            "stronger_l2_regularization": "0.01 across all layers",
            "lower_learning_rate": "0.0002 vs 0.0005",
            "aggressive_early_stopping": "patience=5 vs 8",
            "gradient_clipping": "clipnorm=0.5 vs 1.0",
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "outlier_threshold": args.outlier_threshold,
        },
        "final_metrics": {
            "test_rmse": float(test_rmse),
            "test_r2": float(test_r2),
            "test_mae": float(test_mae),
            "test_bias": float(test_bias),
        },
        "training_epochs_completed": len(history.history["loss"]) if history else 0,
    }

    with open(output_dir / "regularization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\\n=== TRAINING COMPLETED ===")
    print(f"Results saved to: {output_dir}")
    print("Files generated:")
    for file in output_dir.glob("*"):
        if file.is_file():
            print(f"  - {file.name}")


if __name__ == "__main__":
    main()
