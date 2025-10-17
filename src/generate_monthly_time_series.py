#!/usr/bin/env python3
"""
Generate monthly time series plots for ozone concentration data.
This script processes the raw data to create monthly time series visualizations
showing year-over-year progression as shown in the reference image.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from evaluate import monthly_concentration_time_series, decode_month_from_sin_cos


def main():
    # Load the raw data
    data_path = Path("data/input_with_geo_and_interactions_v5.npy")

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file exists.")
        return

    print("Loading raw data...")
    raw_data = np.load(data_path)
    print(f"Raw data shape: {raw_data.shape}")

    # Load column names to find the correct indices
    try:
        with open("data/final_column_names.json", "r") as f:
            column_names = [col.lower() for col in json.load(f)]
    except FileNotFoundError:
        print("Error: final_column_names.json not found")
        return

    # Find column indices
    year_idx = None
    month_idx = None
    ozone_idx = None

    # Define target name variants
    target_name_variants = {
        "ozone": ["ozone", "ozone_concentration"],
    }

    # Find column indices
    year_idx = 2  # Year column

    # Find month_sin and month_cos columns to decode month
    month_sin_idx = None
    month_cos_idx = None

    for i, name in enumerate(column_names):
        if name.lower() == "month_sin":
            month_sin_idx = i
        elif name.lower() == "month_cos":
            month_cos_idx = i

    if month_sin_idx is None or month_cos_idx is None:
        print("Error: Could not find month_sin and month_cos columns")
        return

    print(
        f"Found month_sin at index {month_sin_idx}, month_cos at index {month_cos_idx}"
    )

    # Find ozone column
    try:
        ozone_idx = next(
            i
            for i, name in enumerate(column_names)
            if name in target_name_variants["ozone"]
        )
        print(f"Found ozone column at index {ozone_idx}")
    except StopIteration:
        print("Error: Could not find ozone concentration column")
        return

    # Decode month values from sine/cosine encoding
    print("Decoding month values from sine/cosine encoding...")
    month_sin_vals = raw_data[:, month_sin_idx]
    month_cos_vals = raw_data[:, month_cos_idx]
    month_vals = decode_month_from_sin_cos(month_sin_vals, month_cos_vals)

    print(f"Month values range: {np.min(month_vals)} to {np.max(month_vals)}")
    print(f"Sample decoded months: {month_vals[:10]}")

    # Create a modified data array with year, month, and concentration columns
    # We'll create a simple 3-column array: [year, month, concentration]
    years = raw_data[:, year_idx]
    concentrations = raw_data[:, ozone_idx]

    # Create the data array for the time series function
    time_series_data = np.column_stack([years, month_vals, concentrations])

    # Create output directory
    output_dir = Path("test_results/cnn_lstm_chrono_regularized")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the monthly time series plot
    print("Generating monthly time series plot...")

    save_path = output_dir / "ozone_monthly_time_series.png"

    try:
        fig, axes = monthly_concentration_time_series(
            data=time_series_data,
            year_column=0,  # year is now column 0 in our modified array
            month_column=1,  # month is now column 1 in our modified array
            concentration_column=2,  # concentration is now column 2 in our modified array
            pollutant_name="Ozone",
            save_path=str(save_path),
            show=False,
            figsize=(14, 8),
            start_year=2001,
            end_year=2015,
        )

        print(f"Monthly time series plot saved to: {save_path}")

    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback

        traceback.print_exc()
        return

    print("Monthly time series generation completed successfully!")


if __name__ == "__main__":
    main()
