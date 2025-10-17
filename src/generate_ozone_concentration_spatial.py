#!/usr/bin/env python3
"""
Quick fix script to regenerate ozone concentration map with proper California bounds.
Uses existing trained model from test_results/cnn_lstm_chrono_before/
"""

import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from evaluate import (
    spatial_concentration_map,
    generate_monthly_time_series_from_raw_data,
    plot_truth_prediction_bias_maps,
)
from data_loader import load_data


def main():
    print("=== Quick Map Fix Script ===")
    print("Regenerating ozone concentration map with fixed California bounds...")

    # Paths
    model_path = Path(
        "test_results/cnn_lstm_chrono_regularized/cnn_lstm_regularized_model.keras"
    )
    output_dir = Path("test_results/cnn_lstm_chrono_regularized")
    shapefile_path = Path("data/cb/cb_2018_us_state_20m.shp")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if model exists
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure the model has been trained first.")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load the data for mapping
    print("Loading data for mapping...")
    try:
        # Load raw data and extract coordinates
        raw_data = np.load("data/input_with_geo_and_interactions_v5.npy")
        with open("data/final_column_names.json", "r") as f:
            column_names = [col.lower() for col in json.load(f)]

        # Find coordinate columns
        lon_idx = next(
            i for i, name in enumerate(column_names) if name.lower() == "lon"
        )
        lat_idx = next(
            i for i, name in enumerate(column_names) if name.lower() == "lat"
        )

        # Find ozone column
        target_variants = ["ozone", "ozone_concentration"]
        ozone_idx = next(
            i for i, name in enumerate(column_names) if name in target_variants
        )

        print(
            f"Found columns - Longitude: {lon_idx}, Latitude: {lat_idx}, Ozone: {ozone_idx}"
        )

        # Use a subset of the data for mapping to make it manageable
        subset_size = min(50000, len(raw_data))  # Use up to 50k points for mapping
        indices = np.random.choice(len(raw_data), subset_size, replace=False)

        lons = raw_data[indices, lon_idx]
        lats = raw_data[indices, lat_idx]
        concentrations = raw_data[indices, ozone_idx]

        print(f"✓ Data prepared for mapping - {len(concentrations)} points")
        print(f"Longitude range: {lons.min():.2f} to {lons.max():.2f}")
        print(f"Latitude range: {lats.min():.2f} to {lats.max():.2f}")
        print(
            f"Concentration range: {concentrations.min():.2f} to {concentrations.max():.2f} PPB"
        )

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback

        traceback.print_exc()
        return

    # Note: For this quick fix, we're using raw data for mapping
    # To use model predictions, we'd need to properly handle the sequence structure
    print("Using raw data for spatial mapping (not model predictions)")

    # Generate the fixed concentration map
    print("Generating fixed concentration map...")
    try:
        shapefile_str = str(shapefile_path) if shapefile_path.exists() else None
        if shapefile_str is None:
            print("Warning: Shapefile not found, map will not have state boundaries")

        fig, ax = spatial_concentration_map(
            lons=lons,
            lats=lats,
            concentrations=concentrations,
            pollutant_name="Ozone",
            shapefile_path=shapefile_str,
            save_path=str(output_dir / "ozone_concentration_map_fixed.png"),
            show=False,
            figsize=(10, 8),
            cmap="plasma",
        )

        fig, ax = plot_truth_prediction_bias_maps(
            lons=lons,
            lats=lats,
            concentrations=concentrations,
            pollutant_name="Ozone",
            shapefile_path=shapefile_str,
            save_path=str(output_dir / "ozone_concentration_map_fixed.png"),
            show=False,
            figsize=(10, 8),
            cmap="plasma",
        )

        print(
            f"✓ Fixed concentration map saved to: {output_dir / 'ozone_concentration_map_fixed.png'}"
        )

    except Exception as e:
        print(f"Error generating concentration map: {e}")
        import traceback

        traceback.print_exc()

    # Generate monthly time series if not already present
    monthly_series_path = output_dir / "ozone_monthly_time_series.png"
    if not monthly_series_path.exists():
        print("Generating monthly time series plot...")
        try:
            fig, axes = generate_monthly_time_series_from_raw_data(
                data_path="data/input_with_geo_and_interactions_v5.npy",
                column_names_path="data/final_column_names.json",
                output_dir=str(output_dir),
                pollutant_name="Ozone",
                show=False,
            )

            if fig is not None:
                print(f"✓ Monthly time series plot saved to: {output_dir}")
            else:
                print("✗ Failed to generate monthly time series plot")

        except Exception as e:
            print(f"Error generating monthly time series: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(f"Monthly time series plot already exists at: {monthly_series_path}")

    print("\n=== Fix Complete ===")
    print(f"Check the output directory: {output_dir}")
    print("Files generated:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
