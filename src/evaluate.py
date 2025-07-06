"""
Evaluation module for air pollutant prediction models.

Currently implemented:
- density_scatter_plot: Generate a density scatter (hexbin) of predicted vs. actual values and annotate with R² and RMSE.
- residuals_plot: Generate a residuals plot (Predicted vs. Residuals).

Future functions will include residual plots, feature importance, spatial error maps, etc.
"""

from typing import Tuple, Optional
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import colors as mcolors

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
