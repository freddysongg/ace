from __future__ import annotations

import argparse
import logging
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_loader import load_data, get_field_names
from add_geo_features import add_geospatial_features
from utils import set_global_seed

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature-engineering pipeline for air-pollutant prediction dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/NON-GEO_input_v2_2d.npy"),
        help="Path to the raw .npy file containing the dataset.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/input_with_geo_and_interactions_v4.npy"),
        help="Where to write the processed dataset (.npy).",
    )
    parser.add_argument(
        "--lat-col",
        default="col_1",
        help="Name of the latitude column in the raw data.",
    )
    parser.add_argument(
        "--lon-col",
        default="col_0",
        help="Name of the longitude column in the raw data.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("results/feature_eng"),
        help="Directory in which to save diagnostic plots (will be created).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def main(
    args: argparse.Namespace,
) -> None:  # noqa: C901 – function is intentionally broad
    set_global_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ---------------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------------
    logging.info("Loading raw data from %s", args.input)
    raw = load_data(args.input)
    field_names = get_field_names() or []
    n_cols = raw.shape[1]
    if len(field_names) != n_cols:
        logging.warning(
            "FIELD_NAMES length (%d) does not match data columns (%d). "
            "Padding/truncating with generic labels (col_*) to preserve known names.",
            len(field_names), n_cols,
        )
        if len(field_names) < n_cols:
            field_names = field_names + [f"col_{i}" for i in range(len(field_names), n_cols)]
        else:
            field_names = field_names[:n_cols]

    df = pd.DataFrame(raw, columns=field_names)
    logging.info("Loaded DataFrame with shape %s", df.shape)

    # ---------------------------------------------------------------------
    # 2. Ensure latitude/longitude columns are present
    # ---------------------------------------------------------------------
    if args.lat_col not in df.columns or args.lon_col not in df.columns:
        raise KeyError(
            "Latitude and/or longitude columns not found. "
            "Use --lat-col / --lon-col to specify their names."
        )
    # Standardise column names to `lat` and `lon` for downstream functions
    if args.lat_col != "lat":
        df.rename(columns={args.lat_col: "lat"}, inplace=True)
    if args.lon_col != "lon":
        df.rename(columns={args.lon_col: "lon"}, inplace=True)

    # ---------------------------------------------------------------------
    # 3. Geospatial feature engineering
    # ---------------------------------------------------------------------
    logging.info("Adding geospatial features … this may take a while ⚡")
    tqdm.pandas()
    df = add_geospatial_features(df)
    logging.info("Geospatial features added. New shape: %s", df.shape)

    # ---------------------------------------------------------------------
    # 3b. Temporal & advanced geographic features 
    # ---------------------------------------------------------------------
    logging.info("Adding temporal and advanced geographic features...")

    # 1. Cyclical Month Encoding (robust)
    if "month" in df.columns:
        logging.info("-> Generating cyclical month features (sin/cos)...")
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        # It's good practice to drop the original month column
        df.drop(columns=["month"], inplace=True)

    # 2. Normalized Year
    if "year" in df.columns and "year_normalized" not in df.columns:
        logging.info("-> Generating normalized year feature...")
        scaler = StandardScaler()
        df["year_normalized"] = scaler.fit_transform(df[["year"]])

    # 3. Season One-Hot Encoding
    if "month" in df.columns:
        logging.info("-> Generating season one-hot encoding...")
        seasons = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall",
        }
        df["season"] = df["month"].apply(lambda x: seasons.get(int(x)))
        season_dummies = pd.get_dummies(df["season"], prefix="season", dtype=int)
        df = pd.concat([df, season_dummies], axis=1)
        df.drop(columns=["season"], inplace=True)

    # 4. 3D Geographic Embedding (Cartesian Coordinates)
    if {"lat", "lon"}.issubset(df.columns):
        logging.info("-> Generating 3D geographic coordinate embedding (X, Y, Z)...")
        lat_rad = np.radians(df["lat"])
        lon_rad = np.radians(df["lon"])
        df["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        df["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        df["geo_z"] = np.sin(lat_rad)

    logging.info("Feature engineering enhancements complete. New shape: %s", df.shape)

    # ---------------------------------------------------------------------
    # 4. Manual interaction features (extend as desired)
    # ---------------------------------------------------------------------
    if {"temperature", "elevation"}.issubset(df.columns):
        df["temp_x_elev"] = df["temperature"] * df["elevation"]
    if {"wind_speed", "hour_of_day"}.issubset(df.columns):
        df["wind_x_hour"] = df["wind_speed"] * df["hour_of_day"]
    if {"NO2", "solar_radiation"}.issubset(df.columns):
        df["no2_x_rad"] = df["NO2"] * df["solar_radiation"]

    # ---------------------------------------------------------------------
    # 4b. High-impact interaction features (Phase 1 additions)
    # ---------------------------------------------------------------------
    # The following interactions were identified through domain analysis as
    # having a strong physical basis and high potential explanatory power.
    interaction_specs = {
        # Temperature–Aerosol (secondary OC formation on hot days)
        ("OC", "tasmax_monmean"): "oc_x_tasmax",
        # Soot dispersion by wind
        ("BC", "windspeed_monmean"): "bc_x_windspeed",
        # Nitrate formation favoured by humidity
        ("NOx", "rel_humid_min_monmean"): "nox_x_rhmin",
        # Inland urbanicity vs sea-breeze dilution
        ("urban_fraction", "distance_to_coast_km"): "urbanfrac_x_coastdist",
        # Traffic exhaust trapped in tall urban canyons
        ("road_density_m", "LCZ_13"): "road_dense_x_LCZ13",
        # Rainfall breaks heatwaves – precursor scavenging
        ("tasmax_monmean", "pr_monmean"): "tasmax_x_pr",
        # Elevation effect modulated by latitude (lapse rate variation)
        ("elevation", "lat"): "elev_x_lat",
        # Sulfate formation driven by solar radiation
        ("SO2", "rsds_monmean"): "so2_x_rsds",
    }
    for (c1, c2), new_name in interaction_specs.items():
        if {c1, c2}.issubset(df.columns):
            df[new_name] = df[c1] * df[c2]

    rh_col_candidates = [
        "rel_humid_mean_monmean",
        "rel_humid_mean",
        "rel_humid_monmean",
    ]
    so2_col = "SO2"
    for rh_col in rh_col_candidates:
        if {so2_col, rh_col}.issubset(df.columns):
            df["so2_x_rh"] = df[so2_col] * df[rh_col]
            break 

    extra_cols = [
        "NH3",
        "FRP",
        "modis_hotspot_count",
        "wind700hPa",
        "pbl_height",
        "ENSO_index",
    ]
    present_extra = [c for c in extra_cols if c in df.columns]
    if present_extra:
        logging.info("Retained additional features: %s", ", ".join(present_extra))

    pollutant_cols = [
        "CO",
        "NOx",
        "PM10",
        "TSP",
        "BC",
        "OC",
        "BrC",
        "SO2",
        "PM25",
    ]
    present_pollutants = [c for c in pollutant_cols if c in df.columns]
    if len(present_pollutants) > 2:
        scaler = StandardScaler()
        pollutants_scaled = scaler.fit_transform(df[present_pollutants])
        n_components = min(3, len(present_pollutants))
        pca = PCA(n_components=n_components, random_state=args.seed)
        pollutant_pcs = pca.fit_transform(pollutants_scaled)

        for i in range(n_components):
            df[f"pollutant_pc{i + 1}"] = pollutant_pcs[:, i]

        # Drop original highly correlated pollutant features
        df.drop(columns=present_pollutants, inplace=True)

    # --- Step 2 · Consolidate LCZ features (drop near-zero-variance, plus dense-urban merge)
    lcz_cols = [c for c in df.columns if c.startswith("LCZ_")]
    dense_urban_cols = [c for c in ["LCZ_1", "LCZ_2", "LCZ_3"] if c in df.columns]

    if dense_urban_cols:
        df["LCZ_dense_urban"] = df[dense_urban_cols].sum(axis=1)
        # Remove the individual dense-urban LCZ columns to reduce sparsity
        df.drop(columns=dense_urban_cols, inplace=True)

    # Drop LCZ columns with very little variation (std < 0.01)
    low_var_lcz = [c for c in lcz_cols if c in df.columns and df[c].std() < 0.01]
    if low_var_lcz:
        df.drop(columns=low_var_lcz, inplace=True)

    # --- Step 3 · Simplify humidity & questionable temperature derivatives
    if {"rel_humid_max_monmean", "rel_humid_min_monmean"}.issubset(df.columns):
        df["rel_humid_mean_monmean"] = (
            df["rel_humid_max_monmean"] + df["rel_humid_min_monmean"]
        ) / 2
        df["rel_humid_range_monmean"] = (
            df["rel_humid_max_monmean"] - df["rel_humid_min_monmean"]
        )
        df.drop(
            columns=["rel_humid_max_monmean", "rel_humid_min_monmean"], inplace=True
        )

    if "tasmax_minus_tasmax_minus_tasmin_monmean" in df.columns:
        df.drop(columns=["tasmax_minus_tasmax_minus_tasmin_monmean"], inplace=True)

    if "month" in df.columns and "month_sin" not in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df.drop(columns=["month"], inplace=True)


    # ---------------------------------------------------------------------
    # 5. Diagnostics – correlation heat-map
    # ---------------------------------------------------------------------
    target_cols = [col for col in ["Ozone", "PM2.5", "NO2"] if col in df.columns]
    feature_cols = [c for c in df.columns if c not in target_cols]

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = args.figure_dir / "feature_target_correlation.png"

    logging.info("Generating correlation heat-map → %s", heatmap_path)
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 12))
    corr = df[feature_cols + target_cols].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Feature–Target Correlation (incl. geospatial)")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------------------
    # 6. Save processed dataset
    # ---------------------------------------------------------------------
    final_features = feature_cols  # targets already in correct position later
    df_final = df[final_features + target_cols]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, df_final.values.astype(np.float32))
    logging.info("Saved processed data → %s", args.output)

    logging.info("Feature-engineering pipeline completed ✅")
    
    final_column_names = df_final.columns.tolist()
    column_names_path = args.output.parent / "final_column_names.json"

    with open(column_names_path, 'w') as f:
        json.dump(final_column_names, f, indent=2)

    logging.info("Saved final column names -> %s", column_names_path)
    logging.info("Feature-engineering pipeline completed ✅")


if __name__ == "__main__":
    parser = _build_parser()
    main(parser.parse_args())
