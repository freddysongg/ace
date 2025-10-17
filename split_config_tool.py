#!/usr/bin/env python3
"""
Split Configuration Tool

A utility to help configure, test, and visualize data splitting strategies
for the CNN-LSTM air pollutant prediction system.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.split_strategies import (
    DataSplitter,
    create_chronological_split_config,
    create_random_split_config,
    create_regional_split_config,
    create_rolling_origin_split_config,
    parse_split_config_from_args,
)
from src.data_loader import load_data


def visualize_chronological_split(
    train_data, val_data, test_data, year_col=2, save_path=None
):
    """Visualize chronological split distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Sample counts by year
    all_years = []
    all_labels = []

    if len(train_data) > 0:
        train_years = train_data[:, year_col]
        all_years.extend(train_years)
        all_labels.extend(["Train"] * len(train_years))

    if len(val_data) > 0:
        val_years = val_data[:, year_col]
        all_years.extend(val_years)
        all_labels.extend(["Validation"] * len(val_years))

    if len(test_data) > 0:
        test_years = test_data[:, year_col]
        all_years.extend(test_years)
        all_labels.extend(["Test"] * len(test_years))

    # Create year-wise counts
    unique_years = sorted(set(all_years))
    train_counts = []
    val_counts = []
    test_counts = []

    for year in unique_years:
        train_count = sum(
            1 for y, l in zip(all_years, all_labels) if y == year and l == "Train"
        )
        val_count = sum(
            1 for y, l in zip(all_years, all_labels) if y == year and l == "Validation"
        )
        test_count = sum(
            1 for y, l in zip(all_years, all_labels) if y == year and l == "Test"
        )

        train_counts.append(train_count)
        val_counts.append(val_count)
        test_counts.append(test_count)

    # Stacked bar chart
    width = 0.8
    ax1.bar(unique_years, train_counts, width, label="Train", alpha=0.8)
    ax1.bar(
        unique_years,
        val_counts,
        width,
        bottom=train_counts,
        label="Validation",
        alpha=0.8,
    )
    ax1.bar(
        unique_years,
        test_counts,
        width,
        bottom=[t + v for t, v in zip(train_counts, val_counts)],
        label="Test",
        alpha=0.8,
    )

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Sample Count")
    ax1.set_title("Chronological Split Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Split proportions
    total_train = len(train_data)
    total_val = len(val_data)
    total_test = len(test_data)
    total_samples = total_train + total_val + total_test

    if total_samples > 0:
        proportions = [
            total_train / total_samples,
            total_val / total_samples,
            total_test / total_samples,
        ]
        labels = [
            f"Train\n({total_train:,})",
            f"Validation\n({total_val:,})",
            f"Test\n({total_test:,})",
        ]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        ax2.pie(
            proportions, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
        )
        ax2.set_title("Split Proportions")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def visualize_regional_split(
    train_data, val_data, test_data, lat_col=1, lon_col=0, save_path=None
):
    """Visualize regional split distribution."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot train data
    if len(train_data) > 0:
        ax.scatter(
            train_data[:, lon_col],
            train_data[:, lat_col],
            c="blue",
            alpha=0.6,
            s=1,
            label=f"Train ({len(train_data):,})",
        )

    # Plot validation data
    if len(val_data) > 0:
        ax.scatter(
            val_data[:, lon_col],
            val_data[:, lat_col],
            c="orange",
            alpha=0.6,
            s=1,
            label=f"Validation ({len(val_data):,})",
        )

    # Plot test data
    if len(test_data) > 0:
        ax.scatter(
            test_data[:, lon_col],
            test_data[:, lat_col],
            c="red",
            alpha=0.8,
            s=2,
            label=f"Test ({len(test_data):,})",
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Regional Split Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def print_split_summary(train_data, val_data, test_data, config_dict):
    """Print a summary of the split results."""
    total_samples = len(train_data) + len(val_data) + len(test_data)

    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)

    print(f"Strategy: {config_dict['strategy']}")
    print(f"Total samples: {total_samples:,}")
    print()

    print(
        f"Training set:   {len(train_data):,} samples ({len(train_data)/total_samples*100:.1f}%)"
    )
    print(
        f"Validation set: {len(val_data):,} samples ({len(val_data)/total_samples*100:.1f}%)"
    )
    print(
        f"Test set:       {len(test_data):,} samples ({len(test_data)/total_samples*100:.1f}%)"
    )
    print()

    # Strategy-specific details
    if config_dict["strategy"] == "chronological":
        if "chronological" in config_dict:
            chron_config = config_dict["chronological"]
            if chron_config.get("train_years"):
                print(f"Training years: {chron_config['train_years']}")
                print(f"Validation years: {chron_config.get('val_years', [])}")
                print(f"Test years: {chron_config.get('test_years', [])}")
            else:
                print(
                    f"Training range: {chron_config['train_start']}-{chron_config['train_end']}"
                )
                print(f"Validation year: {chron_config['val_year']}")
                print(
                    f"Test range: {chron_config['test_start']}-{chron_config.get('test_end', 'end')}"
                )

    elif config_dict["strategy"] == "random":
        if "random" in config_dict:
            rand_config = config_dict["random"]
            print(f"Random seed: {rand_config['seed']}")
            print(f"Train fraction: {rand_config['train_frac']}")
            print(f"Validation fraction: {rand_config['val_frac']}")

    elif config_dict["strategy"] == "regional":
        if "regional" in config_dict:
            reg_config = config_dict["regional"]
            bbox = reg_config["test_region_bbox"]
            print(
                f"Test region: ({bbox[0]:.2f}, {bbox[1]:.2f}) to ({bbox[2]:.2f}, {bbox[3]:.2f})"
            )
            print(f"Train/val split method: {reg_config['train_val_split_method']}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Configure and test data splitting strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test default chronological split
  python split_config_tool.py --data data/input_with_geo_and_interactions_v5.npy

  # Test custom chronological split with specific years
  python split_config_tool.py --data data/input_with_geo_and_interactions_v5.npy \\
    --split-strategy chronological --train-years "2001,2002,2003" --val-years "2004" --test-years "2005,2006"

  # Test random split
  python split_config_tool.py --data data/input_with_geo_and_interactions_v5.npy \\
    --split-strategy random --train-frac 0.7 --val-frac 0.2

  # Test regional split
  python split_config_tool.py --data data/input_with_geo_and_interactions_v5.npy \\
    --test-region "-125,32,-115,42" --visualize

  # Save configuration for later use
  python split_config_tool.py --data data/input_with_geo_and_interactions_v5.npy \\
    --split-strategy chronological --save-config my_split_config.json
        """,
    )

    # Data arguments
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the data file (.npy)"
    )
    parser.add_argument(
        "--year-column",
        type=int,
        default=2,
        help="Index of the year column (default: 2)",
    )

    # Split strategy arguments (reuse from main script)
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
        help="Test region bounding box as 'min_lon,min_lat,max_lon,max_lat'",
    )

    # Chronological split options
    parser.add_argument(
        "--train-years", type=str, help="Comma-separated training years"
    )
    parser.add_argument(
        "--val-years", type=str, help="Comma-separated validation years"
    )
    parser.add_argument("--test-years", type=str, help="Comma-separated test years")
    parser.add_argument(
        "--train-start", type=int, default=2001, help="Training start year"
    )
    parser.add_argument("--train-end", type=int, default=2012, help="Training end year")
    parser.add_argument("--val-year", type=int, default=2013, help="Validation year")
    parser.add_argument("--test-start", type=int, default=2014, help="Test start year")
    parser.add_argument("--test-end", type=int, default=2015, help="Test end year")

    # Random split options
    parser.add_argument(
        "--train-frac", type=float, default=0.8, help="Training fraction"
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.1, help="Validation fraction"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output options
    parser.add_argument(
        "--visualize", action="store_true", help="Create visualization of the split"
    )
    parser.add_argument(
        "--save-config", type=str, help="Save split configuration to JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="split_analysis",
        help="Directory to save outputs (default: split_analysis)",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    try:
        data = load_data(args.data)
        print(f"Loaded data with shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Parse split configuration
    try:
        config = parse_split_config_from_args(args)
        splitter = DataSplitter(config)
    except Exception as e:
        print(f"Error creating split configuration: {e}")
        sys.exit(1)

    # Apply split
    print("Applying split strategy...")
    try:
        train_data, val_data, test_data = splitter.split(data)
    except Exception as e:
        print(f"Error applying split: {e}")
        sys.exit(1)

    # Print summary
    config_dict = splitter.get_config_dict()
    print_split_summary(train_data, val_data, test_data, config_dict)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save configuration if requested
    if args.save_config:
        config_path = Path(args.save_config)
        splitter.save_config(config_path)

    # Create visualization if requested
    if args.visualize:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if config_dict["strategy"] in ["chronological", "rolling_origin"]:
            viz_path = output_dir / f"chronological_split_{timestamp}.png"
            visualize_chronological_split(
                train_data,
                val_data,
                test_data,
                year_col=args.year_column,
                save_path=viz_path,
            )

        elif config_dict["strategy"] == "regional":
            viz_path = output_dir / f"regional_split_{timestamp}.png"
            visualize_regional_split(
                train_data, val_data, test_data, save_path=viz_path
            )

        elif config_dict["strategy"] == "random":
            # For random split, show basic distribution
            viz_path = output_dir / f"random_split_{timestamp}.png"
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            sizes = [len(train_data), len(val_data), len(test_data)]
            labels = [
                f"Train\n({len(train_data):,})",
                f"Validation\n({len(val_data):,})",
                f"Test\n({len(test_data):,})",
            ]
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

            ax.pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
            )
            ax.set_title("Random Split Distribution")

            plt.tight_layout()
            plt.savefig(viz_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {viz_path}")

    print(f"\nAnalysis complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
