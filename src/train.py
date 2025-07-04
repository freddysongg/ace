"""
Training module for air pollutant prediction models.

This module contains functions to train:
- Multiple Linear Regression (MLR) models
- CNN+LSTM models (to be implemented)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import re

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    BackupAndRestore,
    TensorBoard,
    ReduceLROnPlateau,
    TerminateOnNaN,
    CSVLogger,
)  # type: ignore

import mlflow
import mlflow.sklearn
import mlflow.keras

from .models import get_mlr_model, get_cnn_lstm_model


def train_mlr_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    pollutant_names: Optional[list] = None,
) -> Dict[str, Dict]:
    """
    Train separate MLR models for each target pollutant.

    Args:
        X_train: Training features array of shape (n_samples, n_features)
        y_train: Training targets array of shape (n_samples, n_pollutants)
        X_val: Optional validation features for reporting validation metrics
        y_val: Optional validation targets for reporting validation metrics
        pollutant_names: Optional list of pollutant names.
                        Defaults to ['Ozone', 'PM2.5', 'NO2']

    Returns:
        Dictionary containing trained models and training metrics for each pollutant:
        {
            'pollutant_name': {
                'model': trained LinearRegression model,
                'train_r2': training R² score,
                'train_rmse': training RMSE,
                'val_r2': validation R² score (if validation data provided),
                'val_rmse': validation RMSE (if validation data provided)
            }
        }
    """
    if pollutant_names is None:
        pollutant_names = ["Ozone", "PM2.5", "NO2"]

    # Ensure y_train is 2D
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)

    n_pollutants = y_train.shape[1]
    if len(pollutant_names) != n_pollutants:
        raise ValueError(
            f"Number of pollutant names ({len(pollutant_names)}) "
            f"doesn't match number of targets ({n_pollutants})"
        )

    trained_models = {}

    # Train a separate model for each pollutant
    for i, pollutant in enumerate(pollutant_names):
        print(f"\nTraining MLR model for {pollutant}...")

        y_train_pollutant = y_train[:, i]

        finite_mask_train = (~np.isnan(y_train_pollutant)) & np.all(
            np.isfinite(X_train), axis=1
        )

        if finite_mask_train.sum() == 0:
            raise ValueError(f"No finite training samples for pollutant {pollutant}.")

        X_train_clean = X_train[finite_mask_train]
        y_train_clean = y_train_pollutant[finite_mask_train]

        model = get_mlr_model()
        model.fit(X_train_clean, y_train_clean)

        y_train_pred = model.predict(X_train_clean)
        train_r2 = r2_score(y_train_clean, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_clean, y_train_pred))

        model_info = {
            "model": model,
            "train_r2": train_r2,
            "train_rmse": train_rmse,
            "n_train": int(finite_mask_train.sum()),
        }

        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Training RMSE: {train_rmse:.4f}")
        print(f"  Training samples used: {finite_mask_train.sum()}")

        if X_val is not None and y_val is not None:
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)

            y_val_pollutant = y_val[:, i]

            finite_mask_val = (~np.isnan(y_val_pollutant)) & np.all(
                np.isfinite(X_val), axis=1
            )

            if finite_mask_val.sum() > 0:
                y_val_pred = model.predict(X_val[finite_mask_val])
                val_r2 = r2_score(y_val_pollutant[finite_mask_val], y_val_pred)
                val_rmse = np.sqrt(
                    mean_squared_error(y_val_pollutant[finite_mask_val], y_val_pred)
                )

                model_info["val_r2"] = val_r2
                model_info["val_rmse"] = val_rmse

                print(f"  Validation R²: {val_r2:.4f}")
                print(f"  Validation RMSE: {val_rmse:.4f}")
            else:
                print("  No finite validation samples – skipping validation metrics.")

        if mlflow is not None and mlflow.active_run() is not None:
            with mlflow.start_run(run_name=f"MLR_{pollutant}", nested=True):
                mlflow.log_param("pollutant", pollutant)
                mlflow.log_param("model_type", "LinearRegression")
                mlflow.log_param("n_train", int(finite_mask_train.sum()))

                mlflow.log_metric("train_r2", train_r2)
                mlflow.log_metric("train_rmse", train_rmse)

                if "val_r2" in model_info:
                    mlflow.log_metric("val_r2", model_info["val_r2"])
                if "val_rmse" in model_info:
                    mlflow.log_metric("val_rmse", model_info["val_rmse"])

                mlflow.sklearn.log_model(model, name="mlr_model")

        trained_models[pollutant] = model_info

    print(f"\nCompleted training {len(trained_models)} MLR models")
    return trained_models


def train_cnn_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    resume: bool = True,
    **kwargs,
) -> Tuple:
    """
    Train a CNN+LSTM model for multi-output air pollutant prediction.

    Args:
        X_train: Training features
        y_train: Training targets (all pollutants)
        X_val: Validation features
        y_val: Validation targets
        epochs: Number of training epochs
        batch_size: Batch size for training
        **kwargs: Additional arguments for model compilation

    Returns:
        Tuple of (trained_model, history)
    """
    if len(X_train.shape) < 3:
        raise ValueError(
            "X_train should be 3-dimensional (samples, timesteps, features) for CNN+LSTM model"
        )

    num_outputs = y_train.shape[1] if len(y_train.shape) > 1 else 1

    model: tf.keras.Model
    if resume:
        ckpt_search_dir = Path("results") / "checkpoints"
        best_ckpt: Optional[Path] = None
        if ckpt_search_dir.exists():
            epoch_ckpts = list(ckpt_search_dir.rglob("epoch*_valLoss*.keras"))
            best_val = float("inf")

            for ckpt in epoch_ckpts:
                match = re.search(r"_valLoss([0-9]+\.[0-9]+)", ckpt.name)
                if match:
                    val = float(match.group(1))
                    if val < best_val:
                        best_val = val
                        best_ckpt = ckpt

            if best_ckpt is None:
                generic_ckpts = list(ckpt_search_dir.rglob("last.keras")) + list(
                    ckpt_search_dir.rglob("*.keras")
                )
                if generic_ckpts:
                    best_ckpt = max(generic_ckpts, key=lambda p: p.stat().st_mtime)

        if best_ckpt is not None:
            print(f"Restoring model weights from checkpoint: {best_ckpt}")
            try:
                model = tf.keras.models.load_model(best_ckpt, compile=False)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(
                    f"Warning: failed to load checkpoint '{best_ckpt}' ({exc}). Rebuilding model instead."
                )
                model = get_cnn_lstm_model(
                    input_shape=X_train.shape[1:], num_outputs=num_outputs
                )
        else:
            model = get_cnn_lstm_model(
                input_shape=X_train.shape[1:], num_outputs=num_outputs
            )
    else:
        print("Resume disabled – starting training from scratch (fresh model).")
        model = get_cnn_lstm_model(
            input_shape=X_train.shape[1:], num_outputs=num_outputs
        )

    compile_kwargs = {
        "optimizer": tf.keras.optimizers.Adam(learning_rate=1e-4),
        "loss": "mse",
        "metrics": [
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
        ],
    }

    compile_kwargs.update(kwargs)

    model.compile(**compile_kwargs)

    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "default"
    ckpt_dir = Path("results") / "checkpoints" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        filepath=str(ckpt_dir / "epoch{epoch:03d}_valLoss{val_loss:.4f}.keras"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    )

    backup_restore_cb = BackupAndRestore(backup_dir=str(ckpt_dir / "backup"))

    tb_log_dir = Path("results") / "logs" / run_id
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_cb = TensorBoard(
        log_dir=str(tb_log_dir),
        update_freq="batch",
        profile_batch=0,
        histogram_freq=1,
    )

    last_ckpt_cb = ModelCheckpoint(
        filepath=str(ckpt_dir / "last.keras"),
        save_best_only=False,
        save_weights_only=False,
    )

    lr_plateau_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6,
    )

    csv_logger_cb = CSVLogger(str(tb_log_dir / "training.csv"))

    callbacks = [
        checkpoint_cb,
        last_ckpt_cb,
        early_stop_cb,
        lr_plateau_cb,
        TerminateOnNaN(),
        backup_restore_cb,
        tensorboard_cb,
        csv_logger_cb,
    ]

    def _finite_mask(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        feat_mask = np.all(np.isfinite(features), axis=(1, 2))
        targ_mask = np.all(np.isfinite(targets), axis=1)
        return feat_mask & targ_mask

    train_mask = _finite_mask(X_train, y_train)
    val_mask = _finite_mask(X_val, y_val)

    if train_mask.sum() == 0:
        raise ValueError(
            "No finite samples available for CNN+LSTM training after filtering."
        )

    if train_mask.sum() < X_train.shape[0]:
        print(
            f"Filtering NaNs/Infs: keeping {train_mask.sum()} / {X_train.shape[0]} training samples"
        )

    if val_mask.sum() == 0:
        raise ValueError(
            "No finite samples available for CNN+LSTM validation after filtering."
        )

    X_train_clean, y_train_clean = X_train[train_mask], y_train[train_mask]
    X_val_clean, y_val_clean = X_val[val_mask], y_val[val_mask]

    history = model.fit(
        X_train_clean,
        y_train_clean,
        validation_data=(X_val_clean, y_val_clean),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if mlflow is not None and mlflow.active_run() is not None:
        with mlflow.start_run(run_name="CNN_LSTM", nested=True):
            mlflow.log_param("model_type", "CNN_LSTM")
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("input_shape", X_train.shape[1:])
            mlflow.log_param("num_outputs", num_outputs)

            for k, v in compile_kwargs.items():
                try:
                    mlflow.log_param(
                        k, v if isinstance(v, (int, float, str, bool)) else str(v)
                    )
                except Exception:
                    mlflow.log_param(k, str(v))

            summary_lines: list[str] = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            mlflow.log_text("\n".join(summary_lines), "model_summary.txt")

            for epoch_idx, _ in enumerate(history.history.get("loss", [])):
                for metric_name, metric_values in history.history.items():
                    mlflow.log_metric(
                        metric_name, metric_values[epoch_idx], step=epoch_idx + 1
                    )

            mlflow.log_artifacts(str(ckpt_dir), artifact_path="checkpoints")
            if tb_log_dir.exists():
                mlflow.log_artifacts(str(tb_log_dir), artifact_path="tensorboard_logs")

            mlflow.keras.log_model(model, name="cnn_lstm_model")

    return model, history
