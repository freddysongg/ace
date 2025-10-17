"""
Evaluation and visualization module for air pollutant prediction models.

Provides comprehensive evaluation functions for model performance assessment,
spatial analysis, and feature importance visualization.
"""

from typing import Tuple, Optional, List, Dict, Any, Union
import os
import json
import warnings

# Optional SHAP for feature importance analysis
try:
    import shap
except ImportError:
    shap = None

import numpy as np
import pandas as pd

# Optional GeoPandas for spatial mapping
try:
    import geopandas as gpd
except ImportError:
    gpd = None
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib import colors as mcolors
from sklearn.inspection import permutation_importance

import mlflow

__all__ = [
    "density_scatter_plot",
    "residuals_plot",
    "feature_importance_bar_chart",
    "spatial_error_map",
    "spatial_concentration_map",  # Added new function
    "training_history_plot",
    "density_scatter_plots_multi",
    "prediction_error_histograms_multi",
    "spatial_error_maps_multi",
    "raw_target_histograms",
    "target_time_series_slice",
    "pred_vs_actual_time_series_slice",
    "plot_keras_evaluation",
    "calculate_summary_metrics",
    "shap_summary_plot",
    "permutation_importance_plot",
    "shap_global_importance_bar_plot",
    "plot_truth_prediction_bias_maps",
    "plot_bias_distribution",
    "monthly_concentration_time_series",
    "decode_month_from_sin_cos",
    "generate_monthly_time_series_from_raw_data",
]


def _log_figure_to_mlflow(fig, artifact_filename: str):
    """Helper function to log figures to MLflow experiment tracking."""
    try:
        if mlflow.active_run() is not None:
            mlflow.log_figure(fig, artifact_filename)
    except Exception:  # pylint: disable=broad-exception-caught
        pass


# -----------------------------------------------------------------------------
# MLR Plots
# -----------------------------------------------------------------------------


def density_scatter_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pollutant_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
    gridsize: int = 100,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (6, 6),
):
    """Generate density scatter plot comparing predicted vs actual values with R² and RMSE metrics.

    Args:
        y_true: Array of true values.
        y_pred: Array of predicted values (same shape as *y_true*).
        pollutant_name: Name of the pollutant (for labels/title).
        save_path: Optional path to save the plot. If *None*, the plot is not saved.
        show: Whether to display the plot via *plt.show()*. Useful in interactive environments.
        gridsize: Hexbin grid size.
        cmap: Colormap for density.
        figsize: Figure size.

    Returns:
        tuple: (*fig*, *ax*) Matplotlib objects for further customization if desired.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape for plotting")

    # Filter out NaN/Inf values for accurate metrics calculation
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        raise ValueError("No finite values available for scatter plot after filtering.")

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Calculate performance metrics for annotation
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Create hexbin density plot with perfect prediction reference line
    fig, ax = plt.subplots(figsize=figsize)
    hb = ax.hexbin(y_true, y_pred, gridsize=gridsize, cmap=cmap, mincnt=1)
    fig.colorbar(hb, ax=ax, label="Counts")

    # Add 1:1 reference line for perfect predictions
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="gray", linewidth=1)

    ax.set_xlabel(f"Actual {pollutant_name}")
    ax.set_ylabel(f"Predicted {pollutant_name}")
    ax.set_title(f"Predicted vs. Actual for {pollutant_name}")

    text_x = 0.05
    text_y = 0.95
    ax.text(
        text_x,
        text_y,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        artifact_name = os.path.basename(save_path)
    else:
        artifact_name = (
            f"{pollutant_name.lower().replace(' ', '_')}_density_scatter.png"
        )

    _log_figure_to_mlflow(fig, artifact_name)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def residuals_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pollutant_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (6, 4),
    point_size: int = 3,
):
    """Generate a residuals plot (Predicted vs. Residuals).

    Args:
        y_true: Array of true values.
        y_pred: Array of predicted values (same shape as *y_true*).
        pollutant_name: Name of the pollutant (for labels/title).
        save_path: Optional path to save the plot. If *None*, the plot is not saved.
        show: Whether to display the plot via *plt.show()*. Useful in interactive environments.
        figsize: Figure size.
        point_size: Marker size.

    Returns:
        tuple: (*fig*, *ax*) Matplotlib objects for further customization if desired.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape for plotting")

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        raise ValueError("No finite values available for residuals plot.")

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        y_pred,
        residuals,
        s=point_size,
        alpha=0.4,
        edgecolor="none",
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=1)

    ax.set_xlabel(f"Predicted {pollutant_name}")
    ax.set_ylabel("Residuals (Predicted - Actual)")
    ax.set_title(f"Residuals Plot for {pollutant_name}")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        artifact_name = os.path.basename(save_path)
    else:
        artifact_name = f"{pollutant_name.lower().replace(' ', '_')}_residuals.png"

    _log_figure_to_mlflow(fig, artifact_name)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def feature_importance_bar_chart(
    model,
    feature_names: list,
    pollutant_name: str,
    top_n: int = 20,
    absolute: bool = True,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (8, 6),
):
    """Generate feature importance bar chart from linear regression coefficients.

    Args:
        model: Trained sklearn LinearRegression model.
        feature_names: List of feature names corresponding to model coefficients.
        pollutant_name: Name of the pollutant (for title).
        top_n: Show top *n* most important features (default 20). If None, show all.
        absolute: If True, ranks features by absolute coefficient magnitude.
        save_path: Optional path to save the plot (PNG).
        show: Whether to display the plot.
        figsize: Matplotlib figure size.

    Returns:
        tuple: (fig, ax) for further customization.
    """
    if not hasattr(model, "coef_"):
        raise AttributeError(
            "Model does not have `coef_` attribute (not a LinearRegression model?)"
        )

    coefs = model.coef_.ravel()  # Ensure 1-D
    if len(coefs) != len(feature_names):
        raise ValueError(
            "Number of coefficients does not match number of feature names"
        )

    importance = np.abs(coefs) if absolute else coefs

    if top_n is not None and top_n < len(feature_names):
        idx = np.argsort(importance)[-top_n:][::-1]
    else:
        idx = np.argsort(importance)[::-1]

    selected_features = [feature_names[i] for i in idx]
    selected_importance = importance[idx]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        range(len(selected_features) - 1, -1, -1),
        selected_importance[::-1],
        color="steelblue",
    )
    ax.set_yticks(range(len(selected_features) - 1, -1, -1))
    ax.set_yticklabels(selected_features[::-1])
    ax.set_xlabel("Absolute Coefficient" if absolute else "Coefficient")
    ax.set_title(f"Feature Importance for {pollutant_name} (MLR)")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        artifact_name = os.path.basename(save_path)
    else:
        artifact_name = (
            f"{pollutant_name.lower().replace(' ', '_')}_feature_importance.png"
        )

    _log_figure_to_mlflow(fig, artifact_name)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def spatial_concentration_map(
    lons: np.ndarray,
    lats: np.ndarray,
    concentrations: np.ndarray,
    pollutant_name: str,
    shapefile_path: "data/cb/cb_2018_us_state_20m.shp",
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "plasma",  # Sequential colormap as required (plasma is more visible)
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    point_size: int = 4,
    alpha: float = 0.7,
    plot_style: str = "hexbin",  # Options: "hexbin", "scatter", "interpolated"
):
    """Generate a spatial concentration scatter map.

    Args:
        lons: 1-D array of longitudes.
        lats: 1-D array of latitudes (must be same length as *lons*).
        concentrations: 1-D array of pollutant concentrations.
        pollutant_name: Name of pollutant for title.
        shapefile_path: Optional path to a shapefile for the map background.
        save_path: Optional path to save the plot.
        show: Whether to display via *plt.show()*.
        figsize: Figure size.
        cmap: Matplotlib colormap (sequential recommended for concentration values).
        vmin: Minimum value for color scale (defaults to min(concentrations)).
        vmax: Maximum value for color scale (defaults to max(concentrations)).
        point_size: Marker size.
        alpha: Marker transparency.
        interpolate: Whether to attempt interpolation for smoother visualization.

    Returns:
        tuple: (fig, ax) Matplotlib objects.
    """
    # Validate input arrays
    if (
        not isinstance(lons, np.ndarray)
        or not isinstance(lats, np.ndarray)
        or not isinstance(concentrations, np.ndarray)
    ):
        raise TypeError("lons, lats, and concentrations must be numpy arrays")

    if not (lons.shape == lats.shape == concentrations.shape):
        raise ValueError(
            f"lons, lats, and concentrations must have the same shape. "
            f"Got lons: {lons.shape}, lats: {lats.shape}, concentrations: {concentrations.shape}"
        )

    # Check for empty arrays
    if lons.size == 0 or lats.size == 0 or concentrations.size == 0:
        raise ValueError("Input arrays cannot be empty")

    # Filter out NaN values
    mask = np.isfinite(concentrations) & np.isfinite(lons) & np.isfinite(lats)
    nan_count = (~mask).sum()
    if nan_count > 0:
        print(
            f"Warning: Filtering out {nan_count} non-finite values ({nan_count/lons.size:.1%} of the data)"
        )

    if mask.sum() == 0:
        raise ValueError(
            "No finite values available for concentration map after filtering NaNs."
        )

    lons_filtered = lons[mask]
    lats_filtered = lats[mask]
    concentrations_filtered = concentrations[mask]

    # Check for reasonable coordinate ranges
    lon_min, lon_max = np.min(lons_filtered), np.max(lons_filtered)
    lat_min, lat_max = np.min(lats_filtered), np.max(lats_filtered)

    if lon_min < -180 or lon_max > 180:
        print(
            f"Warning: Longitude values ({lon_min:.2f}, {lon_max:.2f}) outside typical range (-180 to 180)"
        )

    if lat_min < -90 or lat_max > 90:
        print(
            f"Warning: Latitude values ({lat_min:.2f}, {lat_max:.2f}) outside typical range (-90 to 90)"
        )

    # Check for reasonable concentration values
    conc_min, conc_max = np.min(concentrations_filtered), np.max(
        concentrations_filtered
    )
    if conc_min < 0:
        print(f"Warning: Negative concentration values detected (min: {conc_min:.2f})")

    # === FIX: Use percentiles for a more robust color scale ===
    # Set default vmin/vmax if not provided
    if vmin is None:
        vmin = np.nanpercentile(concentrations_filtered, 2)  # Use 2nd percentile as min
    if vmax is None:
        vmax = np.nanpercentile(
            concentrations_filtered, 98
        )  # Use 98th percentile as max

    # Validate vmin/vmax
    if vmin >= vmax:
        print(
            f"Warning: vmin ({vmin}) >= vmax ({vmax}). Setting vmin to 0 and vmax to max concentration."
        )
        vmin = 0
        vmax = np.nanmax(concentrations_filtered)

    # Use linear normalization instead of TwoSlopeNorm as required
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)

    # Load and plot shapefile background if provided
    map_gdf = None
    if shapefile_path:
        if gpd is None:
            warnings.warn("geopandas is not installed. Skipping map background.")
        elif not os.path.exists(shapefile_path):
            warnings.warn(
                f"Shapefile not found at {shapefile_path}. Skipping map background."
            )
        else:
            # Load the shapefile and filter for California for a tighter map extent
            us_states = gpd.read_file(shapefile_path)
            map_gdf = us_states[us_states["NAME"] == "California"]
            if map_gdf.empty:
                warnings.warn(
                    "Could not find 'California' in the shapefile. Using full extent."
                )
                map_gdf = us_states

            # Plot the shapefile background with better styling
            map_gdf.plot(
                ax=ax,
                edgecolor="black",
                facecolor="white",
                linewidth=1.2,
                alpha=0.3,
                zorder=1,
            )

    # Create smooth interpolated surface if we have enough points
    if len(lons_filtered) > 10:  # Remove upper limit for larger datasets
        try:
            from scipy.interpolate import griddata
            from scipy.ndimage import gaussian_filter

            # Create high-resolution grid for smooth surface
            lon_min, lon_max = np.min(lons_filtered), np.max(lons_filtered)
            lat_min, lat_max = np.min(lats_filtered), np.max(lats_filtered)

            # Extend grid beyond data to ensure full coverage
            lon_pad = (lon_max - lon_min) * 0.1  # Increased padding
            lat_pad = (lat_max - lat_min) * 0.1
            lon_min -= lon_pad
            lon_max += lon_pad
            lat_min -= lat_pad
            lat_max += lat_pad

            # High-resolution grid for smooth surface (like reference image)
            grid_resolution = min(
                120, max(60, int(len(lons_filtered) ** 0.3 * 25))
            )  # Higher resolution
            grid_lon, grid_lat = np.mgrid[
                lon_min : lon_max : complex(grid_resolution),
                lat_min : lat_max : complex(grid_resolution),
            ]

            # Multi-step interpolation for better coverage
            # First pass: linear interpolation
            grid_values = griddata(
                (lons_filtered, lats_filtered),
                concentrations_filtered,
                (grid_lon, grid_lat),
                method="linear",
                fill_value=np.nan,
            )

            # Second pass: fill remaining gaps with nearest neighbor
            nan_mask = np.isnan(grid_values)
            if np.any(nan_mask):
                grid_values_nearest = griddata(
                    (lons_filtered, lats_filtered),
                    concentrations_filtered,
                    (grid_lon, grid_lat),
                    method="nearest",
                    fill_value=np.nan,
                )
                # Only fill NaN areas with nearest neighbor
                grid_values = np.where(nan_mask, grid_values_nearest, grid_values)

            # Light smoothing for professional appearance
            grid_values_smooth = gaussian_filter(
                grid_values, sigma=0.8, mode="constant", cval=np.nan
            )

            # GEOGRAPHIC MASKING - Only show interpolation within state boundaries
            if map_gdf is not None and not map_gdf.empty:
                try:
                    from shapely.geometry import Point
                    from shapely.vectorized import contains

                    california_boundary = map_gdf.unary_union

                    # Use vectorized approach for better performance and accuracy
                    points_x = grid_lon.flatten()
                    points_y = grid_lat.flatten()

                    # Check which points are within California boundaries
                    mask_flat = contains(california_boundary, points_x, points_y)
                    mask = mask_flat.reshape(grid_lon.shape)

                    # Apply mask - set values outside boundaries to NaN
                    grid_values_smooth = np.where(mask, grid_values_smooth, np.nan)

                    print(
                        f"Geographic masking applied: {np.sum(mask)} points inside boundary"
                    )
                except Exception as e:
                    print(f"Geographic masking failed: {e}")
                    # Fallback to original approach
                    try:
                        from shapely.geometry import Point

                        california_boundary = map_gdf.unary_union
                        mask = np.zeros_like(grid_values_smooth, dtype=bool)

                        # Create mask for points within California boundaries
                        for i in range(grid_lon.shape[0]):
                            for j in range(grid_lon.shape[1]):
                                if not np.isnan(grid_values_smooth[i, j]):
                                    point = Point(grid_lon[i, j], grid_lat[i, j])
                                    if california_boundary.contains(
                                        point
                                    ) or california_boundary.touches(point):
                                        mask[i, j] = True

                        # Apply mask - set values outside boundaries to NaN
                        grid_values_smooth = np.where(mask, grid_values_smooth, np.nan)
                    except Exception as fallback_e:
                        print(f"Fallback masking also failed: {fallback_e}")

            # Create smooth surface with many levels for gradient effect
            levels = np.linspace(vmin, vmax, 50)  # Many levels for smooth gradients

            # Main interpolated surface (like reference image)
            contour = ax.contourf(
                grid_lon,
                grid_lat,
                grid_values_smooth,
                levels=levels,
                cmap=cmap,
                norm=norm,
                alpha=0.9,  # Strong surface
                extend="both",
                zorder=1,
            )
            cbar_obj = contour

        except Exception as e:
            print(f"Smooth interpolation failed: {e}")
            # Fallback to scatter plot
            sc = ax.scatter(
                lons_filtered,
                lats_filtered,
                c=concentrations_filtered,
                cmap=cmap,
                norm=norm,
                s=4,  # Slightly larger fallback points
                alpha=0.8,
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )
            cbar_obj = sc
    else:
        # Fallback scatter plot for small datasets
        sc = ax.scatter(
            lons_filtered,
            lats_filtered,
            c=concentrations_filtered,
            cmap=cmap,
            norm=norm,
            s=4,
            alpha=0.8,
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )
        cbar_obj = sc

    # Set appropriate color bar title as required
    cbar = fig.colorbar(
        cbar_obj, ax=ax, orientation="vertical", label="Concentration (PPB)"
    )

    # Add grid lines for better spatial reference
    ax.grid(True, linestyle="--", alpha=0.3)

    # Improve axis labels
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"Spatial Distribution of {pollutant_name} Concentration", fontsize=14)

    # Add data summary in a text box
    stats_text = (
        f"Data points: {len(concentrations_filtered)}\n"
        f"Min: {np.min(concentrations_filtered):.1f} PPB\n"
        f"Max: {np.max(concentrations_filtered):.1f} PPB\n"
        f"Mean: {np.mean(concentrations_filtered):.1f} PPB"
    )
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=10,
    )

    # Add prominent state boundary on top of the data
    if map_gdf is not None and not map_gdf.empty:
        # Plot boundary on top with higher zorder
        map_gdf.plot(
            ax=ax,
            edgecolor="black",
            facecolor="none",
            linewidth=2.0,
            alpha=0.8,
            zorder=10,
        )

        # Set plot limits based on the shapefile's extent with proper padding
        minx, miny, maxx, maxy = map_gdf.total_bounds
        ax_padding_x = (maxx - minx) * 0.05  # Increased padding for full coverage
        ax_padding_y = (maxy - miny) * 0.05
        ax.set_xlim(minx - ax_padding_x, maxx + ax_padding_x)
        ax.set_ylim(miny - ax_padding_y, maxy + ax_padding_y)
    else:
        # If no shapefile, use data extent with proper padding
        data_padding_x = (lons_filtered.max() - lons_filtered.min()) * 0.05
        data_padding_y = (lats_filtered.max() - lats_filtered.min()) * 0.05
        ax.set_xlim(lons_filtered.min() - data_padding_x, lons_filtered.max() + data_padding_x)
        ax.set_ylim(lats_filtered.min() - data_padding_y, lats_filtered.max() + data_padding_y)

    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        artifact_name = os.path.basename(save_path)
    else:
        artifact_name = (
            f"{pollutant_name.lower().replace(' ', '_')}_concentration_map.png"
        )

    _log_figure_to_mlflow(fig, artifact_name)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def training_history_plot(
    history,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Training History (Loss vs. Epochs)",
):
    """Generate a Training History Plot (Loss vs. Epochs).

    Args:
        history: A Keras History object or a dictionary with keys ``"loss"``
            and optionally ``"val_loss"``.
        save_path: Optional path to save the figure (PNG).
        show: If *True*, displays the plot with ``plt.show()``.
        figsize: Matplotlib figure size.
        title: Plot title.

    Returns:
        tuple: (fig, ax) Matplotlib objects.
    """

    if hasattr(history, "history"):
        hist_dict = history.history
    elif isinstance(history, dict):
        hist_dict = history
    else:
        raise TypeError("`history` must be a Keras History object or a dictionary.")

    if "loss" not in hist_dict:
        raise KeyError("`history` does not contain 'loss' key.")

    train_loss = hist_dict["loss"]
    val_loss = hist_dict.get("val_loss", None)

    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, train_loss, "-o", label="Training Loss")
    if val_loss is not None:
        ax.plot(epochs, val_loss, "-o", label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        artifact_name = os.path.basename(save_path)
    else:
        artifact_name = "training_history.png"

    _log_figure_to_mlflow(fig, artifact_name)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def shap_summary_plot(
    shap_values_2d: np.ndarray,
    features_2d: np.ndarray,
    feature_names: list,
    pollutant_name: str,
    max_display: int = 20,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Creates a SHAP summary dot plot (beeswarm) from pre-computed 2D SHAP values.

    Args:
        shap_values_2d (np.ndarray): 2D array of SHAP values (samples, features).
        features_2d (np.ndarray): 2D array of feature values for coloring (samples, features).
        feature_names (list): A list of strings for the feature names.
        pollutant_name (str): The name of the target pollutant.
        max_display (int): Maximum number of features to display in the plot.
        save_path (Optional[str]): The file path to save the plot.
        show (bool): Whether to display the plot.
    """
    if shap_values_2d.ndim != 2 or features_2d.ndim != 2:
        raise ValueError("shap_values_2d and features_2d must be 2D arrays.")

    if len(feature_names) != shap_values_2d.shape[1]:
        raise ValueError("Mismatch between number of feature names and data columns.")

    plt.figure()
    shap.summary_plot(
        shap_values_2d,
        features_2d,
        feature_names=feature_names,
        max_display=max_display,
        plot_type="dot",
        show=False,
    )

    fig = plt.gcf()
    plt.title(f"Feature Importance for {pollutant_name} (SHAP values)")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        _log_figure_to_mlflow(fig, os.path.basename(save_path))

    if show:
        plt.show()
    else:
        plt.close(fig)


def permutation_importance_plot(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    pollutant_name: str,
    n_repeats: int = 10,
    random_state: int = 42,
    top_n: int = 20,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (10, 8),
):
    """Generate a permutation importance plot.

    This function calculates and plots permutation feature importance for a trained model.

    Args:
        model: Trained model with a predict method.
        X: Feature matrix.
        y: Target vector.
        feature_names: List of feature names.
        pollutant_name: Name of the pollutant for plot title.
        n_repeats: Number of times to permute each feature.
        random_state: Random seed for reproducibility.
        top_n: Number of top features to display.
        save_path: Optional path to save the plot.
        show: Whether to display the plot.
        figsize: Figure size.

    Returns:
        tuple: (fig, ax) Matplotlib objects and the permutation importance results.
    """
    # Calculate permutation importance
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state
    )

    # Sort features by importance
    sorted_idx = result.importances_mean.argsort()[::-1]

    # Select top N features
    if top_n is not None and top_n < len(feature_names):
        sorted_idx = sorted_idx[:top_n]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot feature importances
    ax.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=[feature_names[i] for i in sorted_idx],
    )

    ax.set_title(f"Permutation Feature Importance for {pollutant_name}")
    ax.set_xlabel("Decrease in R² score")

    fig.tight_layout()

    # Save plot if path provided
    if save_path:
        fig.savefig(save_path, dpi=300)
        artifact_name = os.path.basename(save_path)
    else:
        artifact_name = (
            f"{pollutant_name.lower().replace(' ', '_')}_permutation_importance.png"
        )

    # Log to MLflow
    _log_figure_to_mlflow(fig, artifact_name)

    # Show or close plot
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, result


def calculate_summary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pollutant_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Calculate comprehensive evaluation metrics for multiple pollutants.

    Args:
        y_true: Array of true values with shape (n_samples, n_pollutants)
        y_pred: Array of predicted values with shape (n_samples, n_pollutants)
        pollutant_names: List of pollutant names corresponding to columns in y_true/y_pred

    Returns:
        Dictionary of metrics for each pollutant with structure:
        {
            'pollutant_name': {
                'R2': float,
                'RMSE': float,
                'MAE': float,
                'Bias': float,
                'NRMSE': float,  # Normalized RMSE (RMSE / range)
                'CV_RMSE': float,  # Coefficient of Variation of RMSE (RMSE / mean)
                'Norm_MAE': float,  # Normalized MAE (MAE / mean)
                'Norm_Bias': float,  # Normalized Bias (Bias / mean)
            }
        }
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    if len(pollutant_names) != y_true.shape[1]:
        raise ValueError(
            f"Number of pollutant names ({len(pollutant_names)}) "
            f"doesn't match number of columns in data ({y_true.shape[1]})"
        )

    metrics = {}

    for i, pollutant in enumerate(pollutant_names):
        # Extract data for this pollutant
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]

        # Filter out NaN values
        mask = np.isfinite(y_true_i) & np.isfinite(y_pred_i)
        if mask.sum() == 0:
            print(f"Warning: No valid data points for {pollutant}")
            metrics[pollutant] = {
                "R2": float("nan"),
                "RMSE": float("nan"),
                "MAE": float("nan"),
                "Bias": float("nan"),
                "NRMSE": float("nan"),
                "CV_RMSE": float("nan"),
                "Norm_MAE": float("nan"),
                "Norm_Bias": float("nan"),
            }
            continue

        y_true_filtered = y_true_i[mask]
        y_pred_filtered = y_pred_i[mask]

        # Calculate basic metrics
        r2 = r2_score(y_true_filtered, y_pred_filtered)
        rmse = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
        mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
        bias = np.mean(y_pred_filtered - y_true_filtered)

        # Calculate normalized metrics
        y_range = np.max(y_true_filtered) - np.min(y_true_filtered)
        y_mean = np.mean(y_true_filtered)

        # Avoid division by zero
        nrmse = rmse / y_range if y_range != 0 else float("nan")
        cv_rmse = rmse / y_mean if y_mean != 0 else float("nan")
        norm_mae = mae / y_mean if y_mean != 0 else float("nan")
        norm_bias = bias / y_mean if y_mean != 0 else float("nan")

        # Store metrics
        metrics[pollutant] = {
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "Bias": bias,
            "NRMSE": nrmse,
            "CV_RMSE": cv_rmse,
            "Norm_MAE": norm_mae,
            "Norm_Bias": norm_bias,
        }

    return metrics


def density_scatter_plots_multi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pollutant_names: List[str],
    save_dir: str,
    show: bool = False,
    gridsize: int = 100,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (6, 6),
):
    """Generate density scatter plots for multiple pollutants.

    Args:
        y_true: Array of true values with shape (n_samples, n_pollutants)
        y_pred: Array of predicted values with shape (n_samples, n_pollutants)
        pollutant_names: List of pollutant names corresponding to columns in y_true/y_pred
        save_dir: Directory to save the plots
        show: Whether to display the plots
        gridsize: Hexbin grid size
        cmap: Colormap for density
        figsize: Figure size

    Returns:
        List of (fig, ax) tuples for each pollutant
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    if len(pollutant_names) != y_true.shape[1]:
        raise ValueError(
            f"Number of pollutant names ({len(pollutant_names)}) "
            f"doesn't match number of columns in data ({y_true.shape[1]})"
        )

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    results = []

    for i, pollutant in enumerate(pollutant_names):
        # Extract data for this pollutant
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]

        # Generate density scatter plot
        save_path = os.path.join(
            save_dir,
            f"{pollutant.lower().replace('.', '').replace(' ', '_')}_density_scatter.png",
        )
        fig, ax = density_scatter_plot(
            y_true_i,
            y_pred_i,
            pollutant_name=pollutant,
            save_path=save_path,
            show=show,
            gridsize=gridsize,
            cmap=cmap,
            figsize=figsize,
        )

        results.append((fig, ax))

    return results


def prediction_error_histograms_multi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pollutant_names: List[str],
    save_dir: str,
    show: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    bins: int = 50,
):
    """Generate prediction error histograms for multiple pollutants.

    Args:
        y_true: Array of true values with shape (n_samples, n_pollutants)
        y_pred: Array of predicted values with shape (n_samples, n_pollutants)
        pollutant_names: List of pollutant names corresponding to columns in y_true/y_pred
        save_dir: Directory to save the plots
        show: Whether to display the plots
        figsize: Figure size
        bins: Number of histogram bins

    Returns:
        List of (fig, ax) tuples for each pollutant
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    if len(pollutant_names) != y_true.shape[1]:
        raise ValueError(
            f"Number of pollutant names ({len(pollutant_names)}) "
            f"doesn't match number of columns in data ({y_true.shape[1]})"
        )

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    results = []

    for i, pollutant in enumerate(pollutant_names):
        # Extract data for this pollutant
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]

        # Filter out NaN values
        mask = np.isfinite(y_true_i) & np.isfinite(y_pred_i)
        if mask.sum() == 0:
            print(f"Warning: No valid data points for {pollutant}")
            continue

        y_true_filtered = y_true_i[mask]
        y_pred_filtered = y_pred_i[mask]

        # Calculate errors
        errors = y_pred_filtered - y_true_filtered

        # Create histogram
        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        n, bins_out, patches = ax.hist(
            errors,
            bins=bins,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )

        # Add vertical line at zero
        ax.axvline(0, color="red", linestyle="--", linewidth=1)

        # Calculate statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        # Add statistics text
        stats_text = (
            f"Mean Error: {mean_error:.4f}\n"
            f"Std Dev: {std_error:.4f}\n"
            f"RMSE: {np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered)):.4f}\n"
            f"MAE: {mean_absolute_error(y_true_filtered, y_pred_filtered):.4f}"
        )

        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Set labels and title
        ax.set_xlabel(f"Prediction Error (Predicted - Actual)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Prediction Error Distribution for {pollutant}")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

        # Save figure
        save_path = os.path.join(
            save_dir,
            f"{pollutant.lower().replace('.', '').replace(' ', '_')}_error_histogram.png",
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)

        # Log to MLflow
        artifact_name = os.path.basename(save_path)
        _log_figure_to_mlflow(fig, artifact_name)

        if show:
            plt.show()
        else:
            plt.close(fig)

        results.append((fig, ax))

    return results


# --- New functions for enhanced visualizations ---


def shap_global_importance_bar_plot(
    shap_values: np.ndarray,
    feature_names: list,
    pollutant_name: str,
    max_display: int = 20,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Creates a global feature importance bar chart from 3D SHAP values.

    Args:
        shap_values (np.ndarray): A 3D numpy array of SHAP values (samples, timesteps, features).
        feature_names (list): A list of strings for the feature names.
        pollutant_name (str): The name of the target pollutant for the plot title.
        max_display (int): The maximum number of features to display.
        save_path (Optional[str]): The file path to save the plot image.
        show (bool): Whether to display the plot interactively.
    """
    if shap_values.ndim != 3:
        raise ValueError(
            f"Expected a 3D SHAP values array, but got shape {shap_values.shape}"
        )

    # Calculate global importance by averaging absolute values across samples and timesteps
    global_importance = np.abs(shap_values).mean(axis=(0, 1))

    # Create a DataFrame for easy sorting and plotting
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": global_importance}
    ).sort_values("importance", ascending=False)

    # Create the plot
    top_n_df = importance_df.head(max_display)
    fig, ax = plt.subplots(figsize=(10, max_display * 0.4))
    ax.barh(top_n_df["feature"], top_n_df["importance"], color="steelblue")
    ax.invert_yaxis()  # Most important feature at the top
    ax.set_xlabel("mean(|SHAP value|) (Average Impact on Model Output)")
    ax.set_title(f"Global Feature Importance for {pollutant_name}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        _log_figure_to_mlflow(fig, os.path.basename(save_path))

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bias_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pollutant_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Creates a box plot showing the distribution of the prediction bias.

    Args:
        y_true (np.ndarray): Array of true target values.
        y_pred (np.ndarray): Array of predicted target values.
        pollutant_name (str): The name of the target pollutant.
        save_path (Optional[str]): The file path to save the plot image.
        show (bool): Whether to display the plot interactively.
    """
    bias = y_pred - y_true
    bias = bias[~np.isnan(bias)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(
        bias,
        vert=False,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
    )
    ax.axvline(0, color="r", linestyle="--", linewidth=1.5, label="Zero Bias")

    mean_bias = np.mean(bias)
    median_bias = np.median(bias)
    std_bias = np.std(bias)

    stats_text = (
        f"Mean: {mean_bias:.3f}\nMedian: {median_bias:.3f}\nStd Dev: {std_bias:.3f}"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_yticks([])
    ax.set_xlabel("Prediction Bias (Predicted - True) [PPB]")
    ax.set_title(f"Distribution of Prediction Bias for {pollutant_name}")
    ax.legend()
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        _log_figure_to_mlflow(fig, os.path.basename(save_path))

    if show:
        plt.show()
    else:
        plt.close(fig)




def plot_truth_prediction_bias_maps(
    lons: np.ndarray,
    lats: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pollutant_name: str,
    shapefile_path: "data/cb/cb_2018_us_state_20m.shp",

    save_path: Optional[str] = None,
    show: bool = False,
    plot_style: str = "hexbin",  # Options: "hexbin", "scatter"
    point_size: int = 4,
):
    """
    Generates a 3-panel plot showing true, predicted, and bias spatial distributions
    with an optional geographic boundary layer.

    Args:
        lons (np.ndarray): Array of longitudes.
        lats (np.ndarray): Array of latitudes.
        y_true (np.ndarray): Array of true target values.
        y_pred (np.ndarray): Array of predicted target values.
        pollutant_name (str): The name of the target pollutant.
        shapefile_path (Optional[str]): Path to a shapefile for the map background.
        save_path (Optional[str]): The file path to save the plot image.
        show (bool): Whether to display the plot interactively.
    """

    def create_simple_plot(
        lons, lats, values, ax, cmap, vmin, vmax, title, map_gdf=None
    ):
        """Helper function to create smooth interpolated surface with geographic masking."""
        import matplotlib.colors as mcolors

        # Create smooth interpolated surface if we have enough points
        if len(lons) > 10:  # Remove upper limit for larger datasets
            try:
                from scipy.interpolate import griddata
                from scipy.ndimage import gaussian_filter

                # Create high-resolution grid for smooth surface
                lon_min, lon_max = np.min(lons), np.max(lons)
                lat_min, lat_max = np.min(lats), np.max(lats)

                # Extend grid beyond data to ensure full coverage
                lon_pad = (lon_max - lon_min) * 0.1  # Increased padding
                lat_pad = (lat_max - lat_min) * 0.1
                lon_min -= lon_pad
                lon_max += lon_pad
                lat_min -= lat_pad
                lat_max += lat_pad

                # High-resolution grid for smooth surface (like reference image)
                grid_resolution = min(
                    120, max(60, int(len(lons) ** 0.3 * 25))
                )  # Higher resolution
                grid_lon, grid_lat = np.mgrid[
                    lon_min : lon_max : complex(grid_resolution),
                    lat_min : lat_max : complex(grid_resolution),
                ]

                # Multi-step interpolation for better coverage
                # First pass: linear interpolation
                grid_values = griddata(
                    (lons, lats),
                    values,
                    (grid_lon, grid_lat),
                    method="linear",
                    fill_value=np.nan,
                )

                # Second pass: fill remaining gaps with nearest neighbor
                nan_mask = np.isnan(grid_values)
                if np.any(nan_mask):
                    grid_values_nearest = griddata(
                        (lons, lats),
                        values,
                        (grid_lon, grid_lat),
                        method="nearest",
                        fill_value=np.nan,
                    )
                    # Only fill NaN areas with nearest neighbor
                    grid_values = np.where(nan_mask, grid_values_nearest, grid_values)

                # Light smoothing for professional appearance
                grid_values_smooth = gaussian_filter(
                    grid_values, sigma=0.8, mode="constant", cval=np.nan
                )

                # GEOGRAPHIC MASKING - Only show interpolation within state boundaries
                if map_gdf is not None and not map_gdf.empty:
                    try:
                        from shapely.geometry import Point
                        from shapely.vectorized import contains

                        california_boundary = map_gdf.unary_union

                        # Use vectorized approach for better performance and accuracy
                        points_x = grid_lon.flatten()
                        points_y = grid_lat.flatten()

                        # Check which points are within California boundaries
                        mask_flat = contains(california_boundary, points_x, points_y)
                        mask = mask_flat.reshape(grid_lon.shape)

                        # Apply mask - set values outside boundaries to NaN
                        grid_values_smooth = np.where(mask, grid_values_smooth, np.nan)

                        print(
                            f"Geographic masking applied for {title}: {np.sum(mask)} points inside boundary"
                        )
                    except Exception as e:
                        print(f"Geographic masking failed for {title}: {e}")
                        # Fallback to original approach
                        try:
                            from shapely.geometry import Point

                            california_boundary = map_gdf.unary_union
                            mask = np.zeros_like(grid_values_smooth, dtype=bool)

                            # Create mask for points within California boundaries
                            for i in range(grid_lon.shape[0]):
                                for j in range(grid_lon.shape[1]):
                                    if not np.isnan(grid_values_smooth[i, j]):
                                        point = Point(grid_lon[i, j], grid_lat[i, j])
                                        if california_boundary.contains(
                                            point
                                        ) or california_boundary.touches(point):
                                            mask[i, j] = True

                            # Apply mask - set values outside boundaries to NaN
                            grid_values_smooth = np.where(
                                mask, grid_values_smooth, np.nan
                            )
                        except Exception as fallback_e:
                            print(
                                f"Fallback masking also failed for {title}: {fallback_e}"
                            )

                # Create smooth surface with many levels for gradient effect
                levels = np.linspace(vmin, vmax, 50)  # Many levels for smooth gradients
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

                # Main interpolated surface (like reference image)
                contour = ax.contourf(
                    grid_lon,
                    grid_lat,
                    grid_values_smooth,
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.9,  # Strong surface
                    extend="both",
                    zorder=1,
                )

                return contour

            except Exception as e:
                print(f"Smooth interpolation failed for {title}: {e}")
                # Fallback to scatter plot
                pass

        # Fallback scatter plot if interpolation fails
        sc = ax.scatter(
            lons,
            lats,
            c=values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=4,  # Slightly larger fallback points
            alpha=0.8,
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )
        return sc

    bias = y_pred - y_true
    map_gdf = None

    if shapefile_path:
        if gpd is None:
            warnings.warn("geopandas is not installed. Skipping map background.")
        elif not os.path.exists(shapefile_path):
            warnings.warn(
                f"Shapefile not found at {shapefile_path}. Skipping map background."
            )
        else:
            try:
                # Load the shapefile and filter for California for proper masking
                us_states = gpd.read_file(shapefile_path)
                map_gdf = us_states[us_states["NAME"] == "California"]
                if map_gdf.empty:
                    warnings.warn(
                        "Could not find 'California' in the shapefile. Using full extent."
                    )
                    map_gdf = us_states
            except Exception as e:
                warnings.warn(
                    f"Failed to load shapefile: {e}. Skipping map background."
                )

    # === FIX: Use percentiles for a more robust color scale ===
    # This prevents outliers from skewing the entire color map.
    vmin = np.nanpercentile(
        np.concatenate([y_true, y_pred]), 2
    )  # Use 2nd percentile as min
    vmax = np.nanpercentile(
        np.concatenate([y_true, y_pred]), 98
    )  # Use 98th percentile as max

    # The bias map should still be centered on zero
    bias_max_abs = np.nanpercentile(np.abs(bias), 98)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
    fig.suptitle(f"Spatial Analysis for {pollutant_name}", fontsize=16)

    point_size = 0.5

    # Panel 1: True Values
    if map_gdf is not None:
        map_gdf.plot(
            ax=axes[0],
            edgecolor="black",
            facecolor="white",
            linewidth=1.0,
            alpha=0.3,
            zorder=1,
        )

    sc1 = create_simple_plot(
        lons, lats, y_true, axes[0], "viridis", vmin, vmax, "True Values", map_gdf
    )
    fig.colorbar(sc1, ax=axes[0], label="True Concentration (PPB)")
    axes[0].set_title("True Values", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    # Add prominent boundary on top
    if map_gdf is not None:
        map_gdf.plot(
            ax=axes[0],
            edgecolor="black",
            facecolor="none",
            linewidth=1.5,
            alpha=0.8,
            zorder=10,
        )

    # Panel 2: Predicted Values
    if map_gdf is not None:
        map_gdf.plot(
            ax=axes[1],
            edgecolor="black",
            facecolor="white",
            linewidth=1.0,
            alpha=0.3,
            zorder=1,
        )

    sc2 = create_simple_plot(
        lons, lats, y_pred, axes[1], "viridis", vmin, vmax, "Predicted Values", map_gdf
    )
    fig.colorbar(sc2, ax=axes[1], label="Predicted Concentration (PPB)")
    axes[1].set_title("Predicted Values", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Longitude")

    # Add prominent boundary on top
    if map_gdf is not None:
        map_gdf.plot(
            ax=axes[1],
            edgecolor="black",
            facecolor="none",
            linewidth=1.5,
            alpha=0.8,
            zorder=10,
        )

    # Panel 3: Bias
    if map_gdf is not None:
        map_gdf.plot(
            ax=axes[2],
            edgecolor="black",
            facecolor="white",
            linewidth=1.0,
            alpha=0.3,
            zorder=1,
        )

    # Use RdBu_r colormap for bias (red=positive bias, blue=negative bias)
    sc3 = create_simple_plot(
        lons,
        lats,
        bias,
        axes[2],
        "RdBu_r",
        -bias_max_abs,
        bias_max_abs,
        "Bias",
        map_gdf,
    )
    fig.colorbar(sc3, ax=axes[2], label="Bias (PPB)")
    axes[2].set_title("Prediction Bias (Pred - True)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Longitude")

    # Add prominent boundary on top
    if map_gdf is not None:
        map_gdf.plot(
            ax=axes[2],
            edgecolor="black",
            facecolor="none",
            linewidth=1.5,
            alpha=0.8,
            zorder=10,
        )

    # Set consistent limits and styling for all panels
    if map_gdf is not None and not map_gdf.empty:
        # Use shapefile bounds for consistent extent
        minx, miny, maxx, maxy = map_gdf.total_bounds
        padding_x = (maxx - minx) * 0.05
        padding_y = (maxy - miny) * 0.05
        xlim = [minx - padding_x, maxx + padding_x]
        ylim = [miny - padding_y, maxy + padding_y]
    else:
        # Use data extent if no shapefile
        padding_x = (lons.max() - lons.min()) * 0.05
        padding_y = (lats.max() - lats.min()) * 0.05
        xlim = [lons.min() - padding_x, lons.max() + padding_x]
        ylim = [lats.min() - padding_y, lats.max() + padding_y]

    for i, ax in enumerate(axes):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")

        # Add subtle grid for better spatial reference
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

        # Improve tick formatting
        ax.tick_params(labelsize=10)

        # Only show y-label on leftmost panel
        if i > 0:
            ax.set_ylabel("")

    # Add overall title
    fig.suptitle(
        f"Spatial Analysis for {pollutant_name}", fontsize=14, fontweight="bold", y=0.95
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.93])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        artifact_name = os.path.basename(save_path)
    else:
        artifact_name = (
            f"{pollutant_name.lower().replace(' ', '_')}_truth_prediction_bias_maps.png"
        )

    _log_figure_to_mlflow(fig, artifact_name)

    if show:
        plt.show()
    else:
        plt.close(fig)




def decode_month_from_sin_cos(month_sin, month_cos):
    """
    Decode month from sine and cosine encoding with 6-month offset correction.
    
    The sine/cosine encoding in this dataset has a 6-month offset from standard encoding.
    This correction ensures that higher ozone values appear in summer months (Jun-Aug)
    rather than winter months (Dec-Feb).
    """
    import math
    # Calculate angle in radians
    angle = np.arctan2(month_sin, month_cos)
    # Convert to month with 6-month offset correction (6 months + base 1 = 7)
    month = ((angle / (2 * math.pi)) * 12) + 7
    # Handle wrap-around for values > 12
    month = np.where(month > 12, month - 12, month)
    # Handle negative angles
    month = np.where(month <= 0, month + 12, month)
    # Round to nearest integer and ensure it's in range 1-12
    month = np.round(month).astype(int)
    month = np.clip(month, 1, 12)
    return month


def monthly_concentration_time_series(
    data: np.ndarray,
    year_column: int,
    month_column: int,
    concentration_column: int,
    pollutant_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (14, 8),
    start_year: int = 2001,
    end_year: int = 2015,
    month_sin_column: Optional[int] = None,
    month_cos_column: Optional[int] = None,
):
    """
    Generate a monthly time series plot showing concentration trends over years.
    
    Args:
        data: Raw data array containing year, month, and concentration columns
        year_column: Index of the year column in data
        month_column: Index of the month column in data (ignored if month_sin/cos_column provided)
        concentration_column: Index of the concentration column in data
        pollutant_name: Name of the pollutant for labels
        save_path: Optional path to save the plot
        show: Whether to display the plot
        figsize: Figure size
        start_year: Start year for analysis
        end_year: End year for analysis
        month_sin_column: Optional index of month_sin column for decoding cyclical month encoding
        month_cos_column: Optional index of month_cos column for decoding cyclical month encoding
    
    Returns:
        tuple: (fig, ax) Matplotlib objects
    """
    # Decode month values from sine/cosine if provided, otherwise use direct month column
    if month_sin_column is not None and month_cos_column is not None:
        # Filter data for valid years and remove NaN values (including month_sin/cos)
        mask = (
            (data[:, year_column] >= start_year) & 
            (data[:, year_column] <= end_year) &
            np.isfinite(data[:, concentration_column]) &
            np.isfinite(data[:, year_column]) &
            np.isfinite(data[:, month_sin_column]) &
            np.isfinite(data[:, month_cos_column])
        )
        
        filtered_data = data[mask]
        
        if len(filtered_data) == 0:
            raise ValueError("No valid data points found for the specified date range")
        
        years = filtered_data[:, year_column].astype(int)
        month_sin_vals = filtered_data[:, month_sin_column]
        month_cos_vals = filtered_data[:, month_cos_column]
        months = decode_month_from_sin_cos(month_sin_vals, month_cos_vals)
        concentrations = filtered_data[:, concentration_column]
    else:
        # Use direct month column (legacy behavior)
        mask = (
            (data[:, year_column] >= start_year) & 
            (data[:, year_column] <= end_year) &
            np.isfinite(data[:, concentration_column]) &
            np.isfinite(data[:, year_column]) &
            np.isfinite(data[:, month_column])
        )
        
        filtered_data = data[mask]
        
        if len(filtered_data) == 0:
            raise ValueError("No valid data points found for the specified date range")
        
        years = filtered_data[:, year_column].astype(int)
        months = filtered_data[:, month_column].astype(int)
        concentrations = filtered_data[:, concentration_column]
    
    # Create datetime-like index for plotting
    import pandas as pd
    
    # Create a DataFrame for easier aggregation
    df = pd.DataFrame({
        'year': years,
        'month': months,
        'concentration': concentrations
    })
    
    # Calculate monthly averages for each year
    monthly_avg = df.groupby(['year', 'month'])['concentration'].agg(['mean', 'std', 'count']).reset_index()
    
    # Create time axis
    monthly_avg['date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=1))
    monthly_avg = monthly_avg.sort_values('date')
    
    # Calculate overall monthly pattern (across all years)
    monthly_pattern = df.groupby('month')['concentration'].agg(['mean', 'std']).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Main time series plot
    dates = monthly_avg['date']
    means = monthly_avg['mean']
    stds = monthly_avg['std']
    counts = monthly_avg['count']
    
    # Plot the main time series with error bands
    ax1.plot(dates, means, 'b-', linewidth=1.5, alpha=0.8, label='Monthly Average')
    
    # Add confidence intervals where we have enough data points
    valid_std_mask = (counts >= 10) & np.isfinite(stds)
    if np.any(valid_std_mask):
        valid_dates = dates[valid_std_mask]
        valid_means = means[valid_std_mask]
        valid_stds = stds[valid_std_mask]
        
        ax1.fill_between(
            valid_dates, 
            valid_means - valid_stds, 
            valid_means + valid_stds,
            alpha=0.2, 
            color='blue',
            label='±1 Std Dev'
        )
    
    # Highlight year-to-year trends
    years_list = sorted(monthly_avg['year'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(years_list)))
    
    for i, year in enumerate(years_list[::3]):  # Show every 3rd year to avoid clutter
        year_data = monthly_avg[monthly_avg['year'] == year]
        if len(year_data) >= 6:  # Only if we have data for at least half the year
            ax1.plot(year_data['date'], year_data['mean'], 
                    color=colors[i*3], alpha=0.6, linewidth=1, 
                    label=f'{year}' if i < 5 else "")  # Only label first 5 years
    
    ax1.set_xlabel('Time (Month/Year)')
    ax1.set_ylabel(f'{pollutant_name} Concentration (PPB)')
    ax1.set_title(f'Monthly {pollutant_name} Concentration ({start_year}-{end_year})')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis to show years clearly
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Seasonal pattern subplot
    months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    ax2.bar(monthly_pattern['month'], monthly_pattern['mean'], 
           yerr=monthly_pattern['std'], capsize=3, 
           alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Month')
    ax2.set_ylabel(f'Avg {pollutant_name} (PPB)')
    ax2.set_title('Seasonal Pattern (Average by Month)')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(months_names, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text box
    overall_mean = np.mean(concentrations)
    overall_std = np.std(concentrations)
    min_conc = np.min(concentrations)
    max_conc = np.max(concentrations)
    n_points = len(concentrations)
    
    stats_text = (
        f'Data Points: {n_points:,}\n'
        f'Mean: {overall_mean:.1f} PPB\n'
        f'Std Dev: {overall_std:.1f} PPB\n'
        f'Range: {min_conc:.1f} - {max_conc:.1f} PPB'
    )
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        artifact_name = os.path.basename(save_path)
    else:
        artifact_name = f"{pollutant_name.lower().replace(' ', '_')}_monthly_time_series.png"
    
    _log_figure_to_mlflow(fig, artifact_name)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig, (ax1, ax2)


def generate_monthly_time_series_from_raw_data(
    data_path: str = "data/input_with_geo_and_interactions_v5.npy",
    column_names_path: str = "data/final_column_names.json",
    output_dir: str = "test_results/monthly_analysis",
    pollutant_name: str = "Ozone",
    show: bool = False,
    start_year: int = 2001,
    end_year: int = 2015,
):
    """
    Generate monthly time series plots from raw data file.
    
    This is a convenience function that replicates the functionality from 
    generate_monthly_time_series.py but can be called directly from evaluate.py.
    
    Args:
        data_path: Path to the raw data .npy file
        column_names_path: Path to the column names JSON file
        output_dir: Directory to save the generated plots
        pollutant_name: Name of the pollutant (default: "Ozone")
        show: Whether to display the plot
        start_year: Start year for analysis
        end_year: End year for analysis
        
    Returns:
        tuple: (fig, axes) if successful, (None, None) if failed
    """
    from pathlib import Path
    import json
    
    # Load the raw data
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return None, None
    
    print("Loading raw data...")
    raw_data = np.load(data_path)
    print(f"Raw data shape: {raw_data.shape}")
    
    # Load column names to find the correct indices
    try:
        with open(column_names_path, "r") as f:
            column_names = [col.lower() for col in json.load(f)]
    except FileNotFoundError:
        print(f"Error: {column_names_path} not found")
        return None, None
    
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
        return None, None
    
    print(f"Found month_sin at index {month_sin_idx}, month_cos at index {month_cos_idx}")
    
    # Find pollutant column
    target_name_variants = {
        "ozone": ["ozone", "ozone_concentration"],
    }
    
    try:
        pollutant_idx = next(
            i for i, name in enumerate(column_names)
            if name in target_name_variants["ozone"]
        )
        print(f"Found {pollutant_name.lower()} column at index {pollutant_idx}")
    except StopIteration:
        print(f"Error: Could not find {pollutant_name.lower()} concentration column")
        return None, None
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the monthly time series plot
    print("Generating monthly time series plot...")
    
    save_path = output_dir / f"{pollutant_name.lower()}_monthly_time_series.png"
    
    try:
        fig, axes = monthly_concentration_time_series(
            data=raw_data,
            year_column=year_idx,
            month_column=0,  # Not used when sin/cos provided
            concentration_column=pollutant_idx,
            pollutant_name=pollutant_name,
            save_path=str(save_path),
            show=show,
            figsize=(14, 8),
            start_year=start_year,
            end_year=end_year,
            month_sin_column=month_sin_idx,
            month_cos_column=month_cos_idx,
        )
        
        print(f"Monthly time series plot saved to: {save_path}")
        return fig, axes
        
    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()
        return None, None
