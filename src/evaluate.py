"""
Evaluation module for air pollutant prediction models.

Currently implemented:
- density_scatter_plot: Generate a density scatter (hexbin) of predicted vs. actual values and annotate with R² and RMSE.
- residuals_plot: Generate a residuals plot (Predicted vs. Residuals).

Future functions will include residual plots, feature importance, spatial error maps, etc.
"""

from typing import Tuple, Optional
import os
import json

try:
    import shap  # type: ignore
except ImportError:  # pragma: no cover – SHAP may be optional at runtime
    shap = None  # pylint: disable=invalid-name

import numpy as np
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
]


def _log_figure_to_mlflow(fig, artifact_filename: str):
    """Log a Matplotlib figure to the active MLflow run (if any).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to log.
    artifact_filename : str
        Filename to use when storing the figure in MLflow artifacts.
    """
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
    """Generate a density scatter plot (hexbin) of predicted vs. actual values.

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

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        raise ValueError("No finite values available for scatter plot after filtering.")

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    fig, ax = plt.subplots(figsize=figsize)
    hb = ax.hexbin(y_true, y_pred, gridsize=gridsize, cmap=cmap, mincnt=1)
    fig.colorbar(hb, ax=ax, label="Counts")

    # Reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="gray", linewidth=1)

    ax.set_xlabel(f"Actual {pollutant_name}")
    ax.set_ylabel(f"Predicted {pollutant_name}")
    ax.set_title(f"Predicted vs. Actual for {pollutant_name}")

    # Annotation with R² and RMSE
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
    point_size: int = 8,
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
    """Generate a feature importance bar chart using model coefficients.

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

    # Compute importance (optionally absolute values)
    importance = np.abs(coefs) if absolute else coefs

    # Select top_n features
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


def spatial_error_map(
    lons: np.ndarray,
    lats: np.ndarray,
    errors: np.ndarray,
    pollutant_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    point_size: int = 20,
    alpha: float = 0.8,
):
    """Generate a spatial error scatter map.

    Args:
        lons: 1-D array of longitudes.
        lats: 1-D array of latitudes (must be same length as *lons*).
        errors: 1-D array of prediction errors (predicted - actual) or absolute errors.
        pollutant_name: Name of pollutant for title.
        save_path: Optional path to save the plot.
        show: Whether to display via *plt.show()*.
        figsize: Figure size.
        cmap: Matplotlib colormap (diverging recommended for signed errors).
        vmin: Minimum value for color scale (defaults to -max|errors|).
        vmax: Maximum value for color scale (defaults to max|errors|).
        point_size: Marker size.
        alpha: Marker transparency.

    Returns:
        tuple: (fig, ax) Matplotlib objects.
    """
    if not (lons.shape == lats.shape == errors.shape):
        raise ValueError("lons, lats, and errors must have the same shape")

    if vmin is None or vmax is None:
        max_abs_err = np.nanmax(np.abs(errors))
        if vmin is None:
            vmin = -max_abs_err
        if vmax is None:
            vmax = max_abs_err

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        lons,
        lats,
        c=errors,
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=alpha,
        edgecolor="k",
        linewidth=0.2,
    )
    cbar = fig.colorbar(sc, ax=ax, orientation="vertical", label="Prediction Error")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial Distribution of Prediction Errors for {pollutant_name}")

    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        artifact_name = os.path.basename(save_path)
    else:
        artifact_name = (
            f"{pollutant_name.lower().replace(' ', '_')}_spatial_error_map.png"
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

    This function plots the training and validation loss curves from a Keras
    *history* object returned by :py-meth:`tensorflow.keras.Model.fit`.

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


# -----------------------------------------------------------------------------
# CNN + LSTM Plots
# -----------------------------------------------------------------------------


def density_scatter_plots_multi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pollutant_names: Optional[list] = None,
    save_dir: Optional[str] = None,
    show: bool = False,
    gridsize: int = 50,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (6, 6),
):
    """Generate density scatter plots for multiple pollutants.

    This is a convenience wrapper around :pyfunc:`density_scatter_plot` that
    iterates over all pollutants, generating one figure per pollutant.

    Args:
        y_true: 2-D array of shape *(n_samples, n_pollutants)* containing ground-truth
            values.
        y_pred: 2-D array with the same shape as *y_true* containing model
            predictions.
        pollutant_names: Optional list of pollutant names. If *None*, defaults to
            generic names ("Pollutant 0", "Pollutant 1", …).
        save_dir: If provided, directory where plots will be saved as
            ``"{pollutant_name}_density_scatter.png"``. The directory is created
            if it does not exist.
        show: Whether to display each plot via ``plt.show()``.
        gridsize: Hexbin grid size (passed through).
        cmap: Matplotlib colormap (passed through).
        figsize: Figure size (passed through).

    Returns:
        dict: Mapping ``pollutant_name → (fig, ax)`` for the generated plots.
    """

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.ndim != 2:
        raise ValueError(
            "y_true and y_pred must be 2-D arrays with shape (n_samples, n_pollutants)"
        )

    n_pollutants = y_true.shape[1]

    if pollutant_names is None:
        pollutant_names = [f"Pollutant {i}" for i in range(n_pollutants)]

    if len(pollutant_names) != n_pollutants:
        raise ValueError(
            "Length of pollutant_names does not match number of columns in y_true/y_pred"
        )

    figs_axes = {}

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for i, name in enumerate(pollutant_names):
        save_path = None
        if save_dir is not None:
            filename = name.lower().replace(" ", "_") + "_density_scatter.png"
            save_path = os.path.join(save_dir, filename)

        fig, ax = density_scatter_plot(
            y_true[:, i],
            y_pred[:, i],
            pollutant_name=name,
            save_path=save_path,
            show=show,
            gridsize=gridsize,
            cmap=cmap,
            figsize=figsize,
        )

        figs_axes[name] = (fig, ax)

    return figs_axes


def prediction_error_histograms_multi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pollutant_names: Optional[list] = None,
    save_dir: Optional[str] = None,
    show: bool = False,
    bins: int = 50,
    figsize: Tuple[int, int] = (6, 4),
    color: str = "steelblue",
):
    """Generate histograms of prediction errors for multiple pollutants.

    Args:
        y_true: 2-D array of ground-truth values *(n_samples, n_pollutants)*.
        y_pred: 2-D array of model predictions (same shape as *y_true*).
        pollutant_names: Optional list of pollutant names.
        save_dir: Directory to save figures. If *None*, figures are not saved.
        show: Whether to display each plot via ``plt.show()``.
        bins: Number of histogram bins.
        figsize: Figure size for each plot.
        color: Bar color.

    Returns:
        dict: Mapping ``pollutant_name → (fig, ax)``.
    """

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.ndim != 2:
        raise ValueError(
            "Inputs must be 2-D arrays with shape (n_samples, n_pollutants)"
        )

    n_pollutants = y_true.shape[1]

    if pollutant_names is None:
        pollutant_names = [f"Pollutant {i}" for i in range(n_pollutants)]

    if len(pollutant_names) != n_pollutants:
        raise ValueError(
            "Length of pollutant_names does not match number of pollutants"
        )

    figs_axes = {}

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    errors = y_pred - y_true

    for i, name in enumerate(pollutant_names):
        save_path = None
        if save_dir is not None:
            filename = name.lower().replace(" ", "_") + "_error_histogram.png"
            save_path = os.path.join(save_dir, filename)

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(errors[:, i], bins=bins, color=color, alpha=0.7, edgecolor="k")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)

        ax.set_xlabel("Prediction Error (Predicted - Actual)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Prediction Error Histogram for {name}")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300)
            artifact_name = os.path.basename(save_path)
        else:
            artifact_name = (
                f"{pollutant_names[i].lower().replace(' ', '_')}_error_histogram.png"
            )

        _log_figure_to_mlflow(fig, artifact_name)

        if show:
            plt.show()
        else:
            plt.close(fig)

        figs_axes[name] = (fig, ax)

    return figs_axes


def spatial_error_maps_multi(
    lons: np.ndarray,
    lats: np.ndarray,
    errors: np.ndarray,
    pollutant_names: Optional[list] = None,
    save_dir: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    point_size: int = 20,
    alpha: float = 0.8,
):
    """Generate spatial error maps for multiple pollutants.

    Args:
        lons: 1-D array of longitudes *(n_samples,)*.
        lats: 1-D array of latitudes *(n_samples,)*.
        errors: 2-D array of prediction errors with shape *(n_samples, n_pollutants)*.
        pollutant_names: Optional list of pollutant names.
        save_dir: Directory to save figures; created if it doesn't exist.
        show: Whether to display each plot.
        figsize, cmap, vmin, vmax, point_size, alpha: Passed through to
            :pyfunc:`spatial_error_map`.

    Returns:
        dict: Mapping ``pollutant_name → (fig, ax)``.
    """

    if errors.ndim != 2:
        raise ValueError(
            "`errors` must be a 2-D array with shape (n_samples, n_pollutants)"
        )

    n_samples, n_pollutants = errors.shape

    if lons.shape[0] != n_samples or lats.shape[0] != n_samples:
        min_len = min(lons.shape[0], lats.shape[0], n_samples)
        if min_len == 0:
            raise ValueError(
                "One of lons/lats has zero length; cannot generate spatial maps."
            )

        import warnings as _warnings

        _warnings.warn(
            "Mismatch between coordinate and error array lengths detected. "
            "Trimming arrays to the shortest common length (" + str(min_len) + ") "
            "to proceed with spatial plotting.",
            RuntimeWarning,
        )

        lons = lons[-min_len:]
        lats = lats[-min_len:]
        errors = errors[-min_len:, :]
        n_samples = min_len

    if pollutant_names is None:
        pollutant_names = [f"Pollutant {i}" for i in range(n_pollutants)]

    if len(pollutant_names) != n_pollutants:
        raise ValueError(
            "Length of pollutant_names does not match number of pollutants"
        )

    figs_axes = {}

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for i, name in enumerate(pollutant_names):
        save_path = None
        if save_dir is not None:
            filename = name.lower().replace(" ", "_") + "_spatial_error_map.png"
            save_path = os.path.join(save_dir, filename)

        fig, ax = spatial_error_map(
            lons,
            lats,
            errors[:, i],
            pollutant_name=name,
            save_path=save_path,
            show=show,
            figsize=figsize,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            point_size=point_size,
            alpha=alpha,
        )

        figs_axes[name] = (fig, ax)

    return figs_axes


# -----------------------------------------------------------------------------
# Initial Data Vis.
# -----------------------------------------------------------------------------


def raw_target_histograms(
    y_raw: np.ndarray,
    pollutant_names: Optional[list] = None,
    save_dir: Optional[str] = None,
    show: bool = False,
    bins: int = 50,
    figsize: Tuple[int, int] = (6, 4),
    color: str = "steelblue",
):
    """Plot histograms of raw (unscaled) target values for each pollutant.

    Args:
        y_raw: 2-D array of raw target values with shape (n_samples, n_pollutants).
        pollutant_names: Optional list of pollutant names corresponding to columns.
        save_dir: Directory to save individual histogram PNGs. If *None*, figures are not saved.
        show: Whether to display the plots via *plt.show()*.
        bins: Number of histogram bins.
        figsize: Matplotlib figure size.
        color: Histogram bar color.
    """
    if y_raw.ndim == 1:
        y_raw = y_raw.reshape(-1, 1)

    n_pollutants = y_raw.shape[1]
    if pollutant_names is None:
        pollutant_names = [f"pollutant_{i}" for i in range(n_pollutants)]

    if len(pollutant_names) != n_pollutants:
        raise ValueError("Length of pollutant_names does not match target columns.")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, name in enumerate(pollutant_names):
        values = y_raw[:, i]
        mask = np.isfinite(values)
        if mask.sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(values[mask], bins=bins, color=color, alpha=0.8)
        ax.set_title(f"{name} – Raw Distribution")
        ax.set_xlabel(name)
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        if save_dir:
            file_path = os.path.join(save_dir, f"{name.replace('.', '')}_histogram.png")
            fig.savefig(file_path, dpi=300)
            _log_figure_to_mlflow(fig, os.path.basename(file_path))

        if show:
            plt.show()
        else:
            plt.close(fig)


def target_time_series_slice(
    y_raw: np.ndarray,
    pollutant_names: Optional[list] = None,
    slice_length: int = 1000,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (10, 4),
):
    """Plot a time-series slice of the first *slice_length* samples of raw targets.

    Args:
        y_raw: 2-D array of raw target values with shape (n_samples, n_pollutants).
        pollutant_names: Optional list of pollutant names.
        slice_length: Number of initial samples to plot.
        save_path: File path to save the combined PNG figure.
        show: Whether to display the plot interactively.
        figsize: Matplotlib figure size.
    """
    if y_raw.ndim == 1:
        y_raw = y_raw.reshape(-1, 1)

    n_pollutants = y_raw.shape[1]
    if pollutant_names is None:
        pollutant_names = [f"pollutant_{i}" for i in range(n_pollutants)]

    if len(pollutant_names) != n_pollutants:
        raise ValueError("Length of pollutant_names does not match target columns.")

    slice_length = min(slice_length, y_raw.shape[0])
    time_axis = np.arange(slice_length)

    fig, ax = plt.subplots(figsize=figsize)
    for i, name in enumerate(pollutant_names):
        ax.plot(time_axis, y_raw[:slice_length, i], label=name, linewidth=1)

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Raw Value")
    ax.set_title(f"Raw Target Time Series – First {slice_length} Samples")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        _log_figure_to_mlflow(fig, os.path.basename(save_path))

    if show:
        plt.show()
    else:
        plt.close(fig)


def pred_vs_actual_time_series_slice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pollutant_names: Optional[list] = None,
    slice_length: int = 500,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (10, 4),
):
    """Plot predicted and actual values for a slice of the test set.

    Args:
        y_true: 2-D array of true values (original scale).
        y_pred: 2-D array of predicted values (original scale), same shape as *y_true*.
        pollutant_names: Optional list of pollutant names.
        slice_length: Number of samples to plot.
        save_path: Where to save the PNG.
        show: Whether to display plot.
        figsize: Figure size.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    n_pollutants = y_true.shape[1]
    if pollutant_names is None:
        pollutant_names = [f"pollutant_{i}" for i in range(n_pollutants)]

    if len(pollutant_names) != n_pollutants:
        raise ValueError("Length of pollutant_names does not match y_true columns")

    slice_length = min(slice_length, y_true.shape[0])
    time_axis = np.arange(slice_length)

    # Create a subplot for each pollutant
    fig, axes = plt.subplots(
        n_pollutants, 1, figsize=(figsize[0], figsize[1] * n_pollutants), sharex=True
    )
    if n_pollutants == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time_axis, y_true[:slice_length, i], label="Actual", linewidth=1)
        ax.plot(time_axis, y_pred[:slice_length, i], label="Predicted", linewidth=1)
        ax.set_ylabel(pollutant_names[i])
        ax.legend()

    axes[-1].set_xlabel("Sample Index")
    fig.suptitle(f"Predicted vs Actual – First {slice_length} Samples")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300)
        _log_figure_to_mlflow(fig, os.path.basename(save_path))

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_keras_evaluation(
    history,
    y_test_orig: np.ndarray,
    y_pred_orig: np.ndarray,
    pollutant_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
    gridsize: int = 60,
    figsize: Tuple[int, int] = (8, 10),
):
    if y_test_orig.shape != y_pred_orig.shape:
        raise ValueError("y_test_orig and y_pred_orig must have the same shape")

    if hasattr(history, "history"):
        hist_dict = history.history
    elif isinstance(history, dict):
        hist_dict = history
    else:
        raise TypeError("history must be a Keras History object or a dict")

    if "mse" not in hist_dict:
        raise KeyError("history does not contain 'mse' key required for RMSE plot")

    train_rmse = [float(np.sqrt(v)) for v in hist_dict["mse"]]
    val_rmse = [float(np.sqrt(v)) for v in hist_dict.get("val_mse", [])]

    epochs = np.arange(1, len(train_rmse) + 1)

    mask = np.isfinite(y_test_orig) & np.isfinite(y_pred_orig)
    if mask.sum() == 0:
        import warnings as _warnings

        _warnings.warn(
            f"plot_keras_evaluation: no finite values for pollutant '{pollutant_name}'. Skipping figure.",
            RuntimeWarning,
        )
        return None, (None, None)

    y_true_f = y_test_orig[mask]
    y_pred_f = y_pred_orig[mask]

    r2 = float(r2_score(y_true_f, y_pred_f))
    rmse_test = float(np.sqrt(mean_squared_error(y_true_f, y_pred_f)))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize)

    ax_top.plot(epochs, train_rmse, "-o", label="Train RMSE")
    if val_rmse:
        ax_top.plot(epochs, val_rmse, "-o", label="Val RMSE")

    ax_top.set_xlabel("Epoch")
    ax_top.set_ylabel("RMSE")
    ax_top.set_title(f"Training vs. Validation RMSE – {pollutant_name}")
    ax_top.legend()
    ax_top.grid(True, linestyle="--", alpha=0.3)

    # (y_true_f, y_pred_f) already prepared with finite mask
    hb = ax_bot.hexbin(
        y_true_f, y_pred_f, gridsize=gridsize, cmap="viridis", mincnt=1, linewidths=0
    )
    fig.colorbar(hb, ax=ax_bot, label="Counts")

    min_val = float(min(y_true_f.min(), y_pred_f.min()))
    max_val = float(max(y_true_f.max(), y_pred_f.max()))
    ax_bot.plot([min_val, max_val], [min_val, max_val], "--", color="gray", linewidth=1)

    ax_bot.set_xlabel(f"Actual {pollutant_name}")
    ax_bot.set_ylabel(f"Predicted {pollutant_name}")
    ax_bot.set_title(f"Predicted vs. Actual – {pollutant_name}")

    text_str = f"R² = {r2:.3f}\nRMSE = {rmse_test:.3f}"
    ax_bot.text(
        0.05,
        0.95,
        text_str,
        transform=ax_bot.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        artifact_name = os.path.basename(save_path)
    else:
        artifact_name = f"evaluation_{pollutant_name.lower().replace(' ', '_')}.png"

    _log_figure_to_mlflow(fig, artifact_name)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax_top, ax_bot)


def calculate_summary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pollutant_names: Optional[list] = None,
) -> dict:
    """Compute summary metrics, including normalized versions, for each pollutant column."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shape")

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    num_pollutants = y_true.shape[1]
    if pollutant_names is None:
        pollutant_names = [f"Pollutant_{i}" for i in range(num_pollutants)]

    all_metrics: dict[str, dict[str, float]] = {}
    for i, name in enumerate(pollutant_names):
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]

        mask = np.isfinite(y_true_col) & np.isfinite(y_pred_col)
        if mask.sum() == 0:
            metrics = {k: float("nan") for k in ["RMSE", "R2", "MAE", "Bias", "NRMSE", "CV_RMSE", "Norm_MAE", "Norm_Bias"]}
        else:
            y_true_f = y_true_col[mask]
            y_pred_f = y_pred_col[mask]

            # --- START: New Normalization Logic ---
            y_range = np.nanmax(y_true_f) - np.nanmin(y_true_f)
            y_mean = np.nanmean(y_true_f)

            raw_rmse = float(np.sqrt(mean_squared_error(y_true_f, y_pred_f)))
            raw_mae = float(mean_absolute_error(y_true_f, y_pred_f))
            raw_bias = float(np.mean(y_pred_f - y_true_f))

            metrics = {
                "RMSE": raw_rmse,
                "R2": float(r2_score(y_true_f, y_pred_f)),
                "MAE": raw_mae,
                "Bias": raw_bias,
            }

            # Calculate normalized metrics, handling potential division by zero
            if y_range > 0:
                metrics["NRMSE"] = raw_rmse / y_range
            else:
                metrics["NRMSE"] = float('nan')

            if y_mean != 0:
                metrics["CV_RMSE"] = raw_rmse / y_mean
                metrics["Norm_MAE"] = raw_mae / y_mean
                metrics["Norm_Bias"] = raw_bias / y_mean
            else:
                metrics["CV_RMSE"] = float('nan')
                metrics["Norm_MAE"] = float('nan')
                metrics["Norm_Bias"] = float('nan')
            # --- END: New Normalization Logic ---

        all_metrics[name] = metrics

    return all_metrics


__all__.append("calculate_summary_metrics")


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # For any other type, try to convert to string
        return str(obj)

def generate_comparison_metrics_summary(
    per_pollutant_metrics: dict,
    multi_output_metrics: dict = None,
    save_path: Optional[str] = None,
    pollutant_names: Optional[list] = None,
) -> dict:
    """Generate summary metrics comparing per-pollutant vs. multi-output approaches.
    
    Parameters
    ----------
    per_pollutant_metrics : dict
        Dictionary with validation and test metrics for per-pollutant models.
        Expected structure: {
            'validation_metrics': {pollutant: {'RMSE': float, 'R2': float, ...}},
            'test_metrics': {pollutant: {'RMSE': float, 'R2': float, ...}}
        }
    multi_output_metrics : dict, optional
        Dictionary with metrics from multi-output model for comparison.
        Same structure as per_pollutant_metrics.
    save_path : str, optional
        Path to save the comparison summary as JSON.
    pollutant_names : list[str], optional
        Names of pollutants to include in comparison.
        
    Returns
    -------
    dict
        Comprehensive comparison summary with improvements and statistics.
    """
    if pollutant_names is None:
        pollutant_names = ["Ozone", "PM2.5", "NO2"]
    
    summary = {
        "per_pollutant_approach": {
            "validation_metrics": per_pollutant_metrics.get("validation_metrics", {}),
            "test_metrics": per_pollutant_metrics.get("test_metrics", {}),
            "aggregate_validation": {},
            "aggregate_test": {}
        },
        "comparison_analysis": {
            "per_pollutant_advantages": [],
            "performance_improvements": {},
            "statistical_summary": {}
        },
        "model_configuration": {
            "approach": "per_pollutant_scaling",
            "num_separate_models": len(pollutant_names),
            "pollutant_specific_configs": {}
        }
    }
    
    # Calculate aggregate metrics for per-pollutant approach
    val_metrics = per_pollutant_metrics.get("validation_metrics", {})
    test_metrics = per_pollutant_metrics.get("test_metrics", {})
    
    if val_metrics:
        for metric_name in ["RMSE", "R2", "MAE", "Bias"]:
            values = [val_metrics[p].get(metric_name, float('nan')) for p in pollutant_names if p in val_metrics]
            if values:
                summary["per_pollutant_approach"]["aggregate_validation"][f"avg_{metric_name.lower()}"] = float(np.nanmean(values))
                summary["per_pollutant_approach"]["aggregate_validation"][f"std_{metric_name.lower()}"] = float(np.nanstd(values))
    
    if test_metrics:
        for metric_name in ["RMSE", "R2", "MAE", "Bias"]:
            values = [test_metrics[p].get(metric_name, float('nan')) for p in pollutant_names if p in test_metrics]
            if values:
                summary["per_pollutant_approach"]["aggregate_test"][f"avg_{metric_name.lower()}"] = float(np.nanmean(values))
                summary["per_pollutant_approach"]["aggregate_test"][f"std_{metric_name.lower()}"] = float(np.nanstd(values))
    
    # Add comparison with multi-output approach if provided
    if multi_output_metrics:
        summary["multi_output_approach"] = {
            "validation_metrics": multi_output_metrics.get("validation_metrics", {}),
            "test_metrics": multi_output_metrics.get("test_metrics", {}),
            "aggregate_validation": {},
            "aggregate_test": {}
        }
        
        # Calculate improvements
        multi_val = multi_output_metrics.get("validation_metrics", {})
        multi_test = multi_output_metrics.get("test_metrics", {})
        
        for pollutant in pollutant_names:
            if pollutant in val_metrics and pollutant in multi_val:
                improvements = {}
                for metric in ["RMSE", "R2", "MAE", "Bias"]:
                    per_val = val_metrics[pollutant].get(metric, float('nan'))
                    multi_val_metric = multi_val[pollutant].get(metric, float('nan'))
                    
                    if not (np.isnan(per_val) or np.isnan(multi_val_metric)):
                        if metric in ["RMSE", "MAE", "Bias"]:  # Lower is better
                            improvement = ((multi_val_metric - per_val) / multi_val_metric) * 100
                        else:  # R2: Higher is better
                            improvement = ((per_val - multi_val_metric) / multi_val_metric) * 100
                        improvements[f"validation_{metric.lower()}_improvement_pct"] = float(improvement)
                
                if improvements:
                    summary["comparison_analysis"]["performance_improvements"][pollutant] = improvements
    
    # Add statistical insights
    if test_metrics:
        rmse_values = [test_metrics[p].get("RMSE", float('nan')) for p in pollutant_names if p in test_metrics]
        r2_values = [test_metrics[p].get("R2", float('nan')) for p in pollutant_names if p in test_metrics]
        
        summary["comparison_analysis"]["statistical_summary"] = {
            "best_rmse_pollutant": pollutant_names[np.nanargmin(rmse_values)] if rmse_values else None,
            "best_r2_pollutant": pollutant_names[np.nanargmax(r2_values)] if r2_values else None,
            "rmse_coefficient_of_variation": float(np.nanstd(rmse_values) / np.nanmean(rmse_values)) if rmse_values else None,
            "r2_range": float(np.nanmax(r2_values) - np.nanmin(r2_values)) if r2_values else None
        }
    
    # Add advantages of per-pollutant approach
    summary["comparison_analysis"]["per_pollutant_advantages"] = [
        "Tailored preprocessing for each pollutant's data distribution",
        "Ozone uses StandardScaler (optimal for small range, no major outliers)",
        "PM2.5 and NO2 use RobustScaler (handles large ranges and outliers)",
        "No log transformation distortion for any pollutant",
        "Independent model optimization per pollutant",
        "Separate model artifacts enable pollutant-specific analysis"
    ]
    
    # Save summary if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Convert to JSON-serializable format before saving
        serializable_summary = convert_to_json_serializable(summary)
        with open(save_path, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        print(f"Comparison metrics summary saved to {save_path}")
    
    return summary


__all__.append("generate_comparison_metrics_summary")


def shap_summary_plot(
    model,
    X_train: np.ndarray,
    feature_names: list,
    save_path: Optional[str] = None,
    show: bool = False,
    background_samples: int = 500,  # Add a new parameter
):
    """Generate and (optionally) save a SHAP summary plot.

    Parameters
    ----------
    model : Any
        Trained model (keras, sklearn, etc.).
    X_train : np.ndarray
        Array of training features (ideally subsampled for speed).
    feature_names : list[str]
        Names of the input features.
    save_path : str, optional
        Path to save the PNG. If *None*, figure is not saved.
    show : bool, default False
        Whether to show the figure interactively.
    background_samples : int, default 500
        Number of samples to use for the background dataset.
    """
    if shap is None:
        raise ImportError(
            "shap package is required for shap_summary_plot but not installed."
        )

    print("Generating SHAP summary plot…")

    # --- START OF MODIFIED BLOCK ---
    # 1. Create a representative background dataset using a random sample
    # This is more robust than just taking the first N samples.
    if X_train.shape[0] > background_samples:
        background_data = X_train[np.random.choice(X_train.shape[0], background_samples, replace=False)]
    else:
        background_data = X_train

    # 2. Use GradientExplainer, which is well-suited for Keras models
    explainer = shap.GradientExplainer(model, background_data)

    # 3. Calculate SHAP values on a larger, separate sample for plotting
    if X_train.shape[0] > 2000:
        # Use up to 2000 samples for the plot itself
        plot_data = X_train[np.random.choice(X_train.shape[0], 2000, replace=False)]
    else:
        plot_data = X_train

    shap_values = explainer.shap_values(plot_data)
    # --- END OF MODIFIED BLOCK ---

    shap_vals_plot = shap_values[0] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(shap_vals_plot, plot_data, feature_names=feature_names, show=show)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        _log_figure_to_mlflow(plt.gcf(), os.path.basename(save_path))

    if not show:
        plt.close()


def _wrap_model_for_permutation(model, output_idx: Optional[int] = None):
    """Wrap a Keras/other model to provide sklearn-compatible *score* method."""

    class _Wrapper:
        def __init__(self, base, idx):
            self.base = base
            self.idx = idx

        def predict(self, X):
            preds = self.base.predict(X)
            if self.idx is not None and preds.ndim == 2:
                preds = preds[:, self.idx]
            return preds

        def score(self, X, y):
            from sklearn.metrics import r2_score

            return r2_score(y, self.predict(X))

    if hasattr(model, "score"):
        return model

    return _Wrapper(model, output_idx)


def permutation_importance_plot(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
    pollutant_name: str,
    save_path: Optional[str] = None,
):
    """Calculate and plot permutation feature importance.

    Works for any model; wraps Keras models to provide *score*.
    """

    print("Calculating permutation importance…")

    sample_pred = model.predict(X_val[:1])
    output_idx = None
    if sample_pred.ndim == 2 and y_val.ndim == 1:
        output_idx = 0

    wrapped_model = _wrap_model_for_permutation(model, output_idx)

    result = permutation_importance(
        wrapped_model,
        X_val,
        y_val,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring="r2",
    )

    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[sorted_idx],
    )
    ax.set_title(f"Permutation Importance for {pollutant_name}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        _log_figure_to_mlflow(fig, os.path.basename(save_path))

    plt.close(fig)
