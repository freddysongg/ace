import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import re
import gc
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available - memory monitoring will be disabled")

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
)

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured memory growth for {len(gpus)} GPU(s)")
except RuntimeError as e:
    print(f"GPU memory configuration failed: {e}")

import mlflow
import mlflow.sklearn
import mlflow.keras

from .models import get_mlr_model, get_cnn_lstm_model, get_mlp_model, get_single_pollutant_mlp_model, get_cnn_lstm_model_no_pooling
from .data_generators import SequenceDataGenerator


def train_mlr_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    pollutant_names: Optional[list] = None,
) -> Dict[str, Dict]:
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


def train_mlp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 32,
    *,
    hidden_units: tuple = (64, 32),
    early_patience: int = 6,
    use_lr_plateau: bool = True,
    lr_plateau_factor: float = 0.5,
    lr_plateau_patience: int = 2,
    **kwargs,
) -> Tuple:
    """Train a simple Multi-Layer Perceptron (fully-connected) model.

    The input is expected to be a 2-D matrix *(samples, features)*.

    Returns
    -------
    model, history
        Trained Keras model and its *History*.
    """

    if X_train.ndim != 2:
        raise ValueError("X_train must be 2-D (samples, features) for MLP training.")

    num_outputs = y_train.shape[1] if y_train.ndim == 2 else 1

    def _finite_mask_2d(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        feat_mask = np.all(np.isfinite(features), axis=1)
        targ_mask = np.all(np.isfinite(targets), axis=1)
        return feat_mask & targ_mask

    train_mask = _finite_mask_2d(X_train, y_train)
    if train_mask.sum() == 0:
        raise ValueError("No finite samples available for MLP training after filtering.")

    X_train_clean = X_train[train_mask]
    y_train_clean = y_train[train_mask]

    if X_val is not None and y_val is not None:
        val_mask = _finite_mask_2d(X_val, y_val)
        X_val_clean = X_val[val_mask]
        y_val_clean = y_val[val_mask]
    else:
        X_val_clean, y_val_clean = None, None

    model = get_mlp_model(
        input_dim=X_train.shape[1],
        num_outputs=num_outputs,
        hidden_units=hidden_units,
    )

    compile_kwargs = {
        "optimizer": tf.keras.optimizers.Adam(learning_rate=5e-5),
        "loss": "mse",
        "metrics": [
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
        ],
    }
    compile_kwargs.update(kwargs)
    model.compile(**compile_kwargs)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=early_patience,
            restore_best_weights=True,
        ),
        TerminateOnNaN(),
    ]

    if use_lr_plateau:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=lr_plateau_factor,
                patience=lr_plateau_patience,
                min_lr=1e-6,
                verbose=1,
            )
        )

    history = model.fit(
        X_train_clean,
        y_train_clean,
        validation_data=(X_val_clean, y_val_clean) if X_val_clean is not None and y_val_clean is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if mlflow is not None and mlflow.active_run() is not None:
        with mlflow.start_run(run_name="MLP", nested=True):
            mlflow.log_param("model_type", "MLP")
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("input_dim", X_train.shape[1])
            mlflow.log_param("num_outputs", num_outputs)
            mlflow.log_param("hidden_units", ",".join(map(str, hidden_units)))
            mlflow.log_param("early_patience", early_patience)
            mlflow.log_param("lr_plateau", use_lr_plateau)

            for k, v in compile_kwargs.items():
                try:
                    mlflow.log_param(k, v if isinstance(v, (int, float, str, bool)) else str(v))
                except Exception:
                    mlflow.log_param(k, str(v))

            for epoch_idx, _ in enumerate(history.history.get("loss", [])):
                for metric_name, metric_values in history.history.items():
                    mlflow.log_metric(metric_name, metric_values[epoch_idx], step=epoch_idx + 1)

            mlflow.keras.log_model(model, name="mlp_model")

    return model, history


def train_single_pollutant_mlp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    pollutant_name: str = "Unknown",
    epochs: int = 100,
    batch_size: int = 32,
    *,
    hidden_units: tuple = (64, 32),
    dropout: float = 0.3,
    l2_reg: Optional[float] = None,
    early_patience: int = 6,
    use_lr_plateau: bool = True,
    lr_plateau_factor: float = 0.5,
    lr_plateau_patience: int = 2,
    **kwargs,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Train a single-pollutant MLP model for air quality prediction.
    
    This function handles 1D target arrays and single-output model training,
    with integrated MLflow logging for individual pollutant models.
    
    Args:
        X_train: Training features array of shape (n_samples, n_features)
        y_train: Training targets array of shape (n_samples,) - 1D for single pollutant
        X_val: Optional validation features for early stopping and monitoring
        y_val: Optional validation targets for early stopping and monitoring
        pollutant_name: Name of the pollutant being trained (for logging)
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        hidden_units: Tuple of hidden layer sizes
        dropout: Dropout rate for regularization
        l2_reg: L2 regularization strength
        early_patience: Patience for early stopping
        use_lr_plateau: Whether to use learning rate reduction on plateau
        lr_plateau_factor: Factor to reduce learning rate by
        lr_plateau_patience: Patience for learning rate reduction
        **kwargs: Additional arguments for model compilation
        
    Returns:
        Tuple of (trained_model, history)
        
    Raises:
        ValueError: If no finite samples are available for training
        RuntimeError: If model training fails
    """
    
    if X_train.ndim != 2:
        raise ValueError("X_train must be 2-D (samples, features) for MLP training.")
    
    if y_train.ndim == 2:
        if y_train.shape[1] != 1:
            raise ValueError(f"Expected single pollutant target (1D or shape (n, 1)), got shape {y_train.shape}")
        y_train = y_train.flatten()
    elif y_train.ndim != 1:
        raise ValueError(f"y_train must be 1D for single pollutant training, got shape {y_train.shape}")
    
    print(f"\nTraining single-pollutant MLP model for {pollutant_name}...")
    print(f"Input shape: {X_train.shape}, Target shape: {y_train.shape}")
    
    def _finite_mask_1d(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Create mask for finite samples with 1D targets."""
        feat_mask = np.all(np.isfinite(features), axis=1)
        targ_mask = np.isfinite(targets)
        return feat_mask & targ_mask

    train_mask = _finite_mask_1d(X_train, y_train)
    if train_mask.sum() == 0:
        raise ValueError(f"No finite samples available for {pollutant_name} MLP training after filtering.")
    
    if train_mask.sum() < X_train.shape[0]:
        print(f"Filtering NaNs/Infs: keeping {train_mask.sum()} / {X_train.shape[0]} training samples")

    X_train_clean = X_train[train_mask]
    y_train_clean = y_train[train_mask]

    X_val_clean, y_val_clean = None, None
    if X_val is not None and y_val is not None:
        if y_val.ndim == 2:
            if y_val.shape[1] != 1:
                raise ValueError(f"Expected single pollutant validation target (1D or shape (n, 1)), got shape {y_val.shape}")
            y_val = y_val.flatten()
        elif y_val.ndim != 1:
            raise ValueError(f"y_val must be 1D for single pollutant training, got shape {y_val.shape}")
            
        val_mask = _finite_mask_1d(X_val, y_val)
        if val_mask.sum() > 0:
            X_val_clean = X_val[val_mask]
            y_val_clean = y_val[val_mask]
            print(f"Validation samples: {val_mask.sum()} / {X_val.shape[0]}")
        else:
            print("Warning: No finite validation samples available")

    try:
        model = get_single_pollutant_mlp_model(
            input_dim=X_train.shape[1],
            hidden_units=hidden_units,
            dropout=dropout,
            l2_reg=l2_reg,
        )

        compile_kwargs = {
            "optimizer": tf.keras.optimizers.Adam(learning_rate=5e-5),
            "loss": "mse",
            "metrics": [
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
            ],
        }
        compile_kwargs.update(kwargs)
        model.compile(**compile_kwargs)

        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val_clean is not None else "loss",
                patience=early_patience,
                restore_best_weights=True,
            ),
            TerminateOnNaN(),
        ]

        if use_lr_plateau:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor="val_loss" if X_val_clean is not None else "loss",
                    factor=lr_plateau_factor,
                    patience=lr_plateau_patience,
                    min_lr=1e-6,
                    verbose=1,
                )
            )

        print(f"Starting training for {pollutant_name}...")
        history = model.fit(
            X_train_clean,
            y_train_clean,
            validation_data=(X_val_clean, y_val_clean) if X_val_clean is not None and y_val_clean is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        print(f"Training completed for {pollutant_name}")
        
        if mlflow is not None and mlflow.active_run() is not None:
            with mlflow.start_run(run_name=f"MLP_{pollutant_name}", nested=True):
                mlflow.log_param("pollutant", pollutant_name)
                mlflow.log_param("model_type", "Single_Pollutant_MLP")
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("input_dim", X_train.shape[1])
                mlflow.log_param("num_outputs", 1)
                mlflow.log_param("hidden_units", ",".join(map(str, hidden_units)))
                mlflow.log_param("dropout", dropout)
                mlflow.log_param("l2_reg", l2_reg)
                mlflow.log_param("early_patience", early_patience)
                mlflow.log_param("lr_plateau", use_lr_plateau)
                mlflow.log_param("n_train_samples", len(X_train_clean))
                if X_val_clean is not None:
                    mlflow.log_param("n_val_samples", len(X_val_clean))

                for k, v in compile_kwargs.items():
                    try:
                        mlflow.log_param(k, v if isinstance(v, (int, float, str, bool)) else str(v))
                    except Exception:
                        mlflow.log_param(k, str(v))

                for epoch_idx, _ in enumerate(history.history.get("loss", [])):
                    for metric_name, metric_values in history.history.items():
                        mlflow.log_metric(metric_name, metric_values[epoch_idx], step=epoch_idx + 1)

                final_loss = history.history.get("loss", [])[-1] if history.history.get("loss") else None
                if final_loss is not None:
                    mlflow.log_metric("final_train_loss", final_loss)
                
                if X_val_clean is not None:
                    final_val_loss = history.history.get("val_loss", [])[-1] if history.history.get("val_loss") else None
                    if final_val_loss is not None:
                        mlflow.log_metric("final_val_loss", final_val_loss)

                sanitized_name = pollutant_name.lower().replace(".", "").replace(" ", "_")
                mlflow.keras.log_model(model, name=f"single_pollutant_mlp_{sanitized_name}")

        return model, history
        
    except Exception as e:
        error_msg = f"Training failed for {pollutant_name}: {str(e)}"
        print(f"ERROR: {error_msg}")
        
        if mlflow is not None and mlflow.active_run() is not None:
            with mlflow.start_run(run_name=f"MLP_{pollutant_name}_FAILED", nested=True):
                mlflow.log_param("pollutant", pollutant_name)
                mlflow.log_param("model_type", "Single_Pollutant_MLP")
                mlflow.log_param("training_status", "FAILED")
                mlflow.log_param("error_message", str(e))
        
        raise RuntimeError(error_msg) from e


def train_cnn_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    resume: bool = True,
    model_builder: Optional[callable] = None,
    use_generator: bool = True,
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
    if use_generator:
        if len(X_train.shape) != 2:
            raise ValueError(
                "X_train should be 2-dimensional (samples, features) when using generator mode"
            )
        lookback = 7
        input_shape = (lookback, X_train.shape[1])
    else:
        if len(X_train.shape) < 3:
            raise ValueError(
                "X_train should be 3-dimensional (samples, timesteps, features) for CNN+LSTM model"
            )
        if X_train.shape[2] == 1:
            raise ValueError(
                "Detected feature dimension of size 1 for CNN+LSTM input. This "
                "suggests the input was reshaped incorrectly. Build proper look-back "
                "sequences via 'prepare_sequences.py' (or data_loader.create_lookback_sequences) "
                "so inputs have shape (samples, timesteps, features)."
            )
        input_shape = X_train.shape[1:]

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
                builder_fn = model_builder or get_cnn_lstm_model
                model = builder_fn(
                    input_shape=input_shape, num_outputs=num_outputs
                )
        else:
            builder_fn = model_builder or get_cnn_lstm_model
            model = builder_fn(
                input_shape=input_shape, num_outputs=num_outputs
            )
    else:
        print("Resume disabled - starting training from scratch (fresh model).")
        builder_fn = model_builder or get_cnn_lstm_model
        model = builder_fn(
            input_shape=input_shape, num_outputs=num_outputs
        )

    compile_kwargs = {
        "optimizer": tf.keras.optimizers.Adam(learning_rate=5e-5),
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
    
    if PSUTIL_AVAILABLE:
        class MemoryMonitorCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.process = psutil.Process()
                
            def on_epoch_end(self, epoch, logs=None):
                memory_info = self.process.memory_info()
                memory_gb = memory_info.rss / 1024**3
                print(f"Epoch {epoch + 1}: Memory usage: {memory_gb:.2f} GB")
                if memory_gb > 12: 
                    print(f"WARNING: High memory usage detected: {memory_gb:.2f} GB")
                    gc.collect()  
        
        memory_monitor_cb = MemoryMonitorCallback()
    else:
        memory_monitor_cb = None

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
    
    if memory_monitor_cb is not None:
        callbacks.append(memory_monitor_cb)

    if use_generator:
        def _finite_mask_2d(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
            feat_mask = np.all(np.isfinite(features), axis=1)
            targ_mask = np.all(np.isfinite(targets), axis=1)
            return feat_mask & targ_mask

        train_mask = _finite_mask_2d(X_train, y_train)
        val_mask = _finite_mask_2d(X_val, y_val)

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

        train_generator = SequenceDataGenerator(
            X_train_clean, y_train_clean,
            lookback=lookback,
            batch_size=batch_size,
            shuffle=True
        )
        val_generator = SequenceDataGenerator(
            X_val_clean, y_val_clean,
            lookback=lookback,
            batch_size=batch_size,
            shuffle=False
        )

        print(f"Training with generator: {len(train_generator)} batches per epoch")
        print(f"Validation with generator: {len(val_generator)} batches per epoch")
        
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_info = process.memory_info()
            print(f"Memory usage before training: {memory_info.rss / 1024**3:.2f} GB")
        
        gc.collect()

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
    else:
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
            mlflow.log_param("input_shape", input_shape)
            mlflow.log_param("use_generator", use_generator)
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


def train_simple_lstm_model(
    X_train: "np.ndarray",  # type: ignore
    y_train: "np.ndarray",  # type: ignore
    X_val: "np.ndarray",  # type: ignore
    y_val: "np.ndarray",  # type: ignore,
    epochs: int = 100,
    batch_size: int = 32,
    **kwargs,
):
    """Train a *single*-layer LSTM sequence model.

    This is a thin wrapper around :func:`train_cnn_lstm_model` that
    passes :func:`~src.models.get_simple_lstm_model` as the `model_builder`.
    All arguments (other than `model_builder`) are forwarded unchanged.
    """

    from .models import get_simple_lstm_model

    return train_cnn_lstm_model(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        model_builder=get_simple_lstm_model,
        **kwargs,
    )


def train_single_pollutant_cnn_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pollutant_name: str = "Pollutant",
    epochs: int = 50,
    batch_size: int = 64,
    resume: bool = True,
    use_generator: bool = False,
    **kwargs,
) -> Tuple:
    """
    Train a CNN+LSTM model for single-pollutant prediction.
    
    This function is specifically designed for per-pollutant training,
    similar to train_single_pollutant_mlp_model but for CNN+LSTM architecture.
    
    Args:
        X_train: Training sequences with shape (samples, timesteps, features)
        y_train: Training targets for single pollutant (1D array)
        X_val: Validation sequences
        y_val: Validation targets for single pollutant (1D array)
        pollutant_name: Name of the pollutant being trained
        epochs: Number of training epochs
        batch_size: Training batch size
        resume: Whether to resume from checkpoint
        use_generator: Whether to use data generator (False for pre-built sequences)
        **kwargs: Additional arguments for model compilation
    
    Returns:
        Tuple of (trained_model, history)
    """
    print(f"Training single-pollutant CNN+LSTM model for {pollutant_name}")
    print(f"Input shape: {X_train.shape}, Target shape: {y_train.shape}")
    
    if y_train.ndim > 1:
        if y_train.shape[1] == 1:
            y_train = y_train.ravel()
            y_val = y_val.ravel()
        else:
            raise ValueError(f"Expected single-pollutant targets, got shape {y_train.shape}")
    
    if use_generator:
        if len(X_train.shape) != 2:
            raise ValueError(
                "X_train should be 2-dimensional (samples, features) when using generator mode"
            )
        lookback = 7
        input_shape = (lookback, X_train.shape[1])
    else:
        if len(X_train.shape) < 3:
            raise ValueError(
                "X_train should be 3-dimensional (samples, timesteps, features) for CNN+LSTM model"
            )
        if X_train.shape[2] == 1:
            raise ValueError(
                "Detected feature dimension of size 1 for CNN+LSTM input. Build proper look-back "
                "sequences so inputs have shape (samples, timesteps, features)."
            )
        input_shape = X_train.shape[1:]
    
    num_outputs = 1
    
    model: tf.keras.Model
    if resume:
        ckpt_search_dir = Path("results") / "checkpoints" / f"cnn_lstm_{pollutant_name.lower().replace('.', '').replace(' ', '_')}"
        best_ckpt: Optional[Path] = None
        if ckpt_search_dir.exists():
            epoch_ckpts = list(ckpt_search_dir.rglob("epoch*_valLoss*.keras"))
            if epoch_ckpts:
                def extract_val_loss(ckpt_path: Path) -> float:
                    match = re.search(r"valLoss([\d.]+)", ckpt_path.name)
                    return float(match.group(1)) if match else float('inf')
                
                best_ckpt = min(epoch_ckpts, key=extract_val_loss)
                print(f"Found checkpoint for {pollutant_name}: {best_ckpt}")
        
        if best_ckpt and best_ckpt.exists():
            print(f"Resuming {pollutant_name} training from checkpoint: {best_ckpt}")
            model = tf.keras.models.load_model(best_ckpt)
        else:
            print(f"No valid checkpoint found for {pollutant_name}, creating new model")
            model = get_cnn_lstm_model_no_pooling(input_shape, num_outputs)
    else:
        print(f"Creating new CNN+LSTM model for {pollutant_name}")
        print(f"DEBUG: Using get_cnn_lstm_model_no_pooling with input_shape={input_shape}, num_outputs={num_outputs}")
        model = get_cnn_lstm_model_no_pooling(input_shape, num_outputs)
        print(f"DEBUG: Created model with name: {model.name}")
    
    compile_kwargs = {
        "optimizer": "adam",
        "loss": "mse",
        "metrics": ["mse", "mae"],
    }
    compile_kwargs.update(kwargs)
    model.compile(**compile_kwargs)
    
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "default"
    ckpt_dir = Path("results") / "checkpoints" / f"cnn_lstm_{pollutant_name.lower().replace('.', '').replace(' ', '_')}" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(ckpt_dir / "epoch{epoch:03d}_valLoss{val_loss:.4f}.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        TerminateOnNaN(),
        CSVLogger(str(ckpt_dir / f"training_log_{pollutant_name.lower().replace('.', '').replace(' ', '_')}.csv")),
    ]
    
    if PSUTIL_AVAILABLE:
        class MemoryMonitorCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.process = psutil.Process()
                
            def on_epoch_end(self, epoch, logs=None):
                memory_info = self.process.memory_info()
                memory_gb = memory_info.rss / 1024**3
                print(f"Epoch {epoch + 1} - Memory usage: {memory_gb:.2f} GB")
                
                if mlflow.active_run():
                    mlflow.log_metric(f"memory_gb_{pollutant_name.lower().replace('.', '').replace(' ', '_')}", memory_gb, step=epoch)
        
        callbacks.append(MemoryMonitorCallback())
    
    def _finite_mask_1d(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Create mask for finite samples with 1D targets."""
        if features.ndim == 3:  
            feat_mask = np.all(np.isfinite(features), axis=(1, 2))
        else:
            feat_mask = np.all(np.isfinite(features), axis=1)
        targ_mask = np.isfinite(targets)
        return feat_mask & targ_mask
    
    train_mask = _finite_mask_1d(X_train, y_train)
    X_train_clean = X_train[train_mask]
    y_train_clean = y_train[train_mask]
    
    val_mask = _finite_mask_1d(X_val, y_val)
    X_val_clean = X_val[val_mask]
    y_val_clean = y_val[val_mask]
    
    print(f"Training samples for {pollutant_name}: {len(X_train_clean)} (filtered from {len(X_train)})")
    print(f"Validation samples for {pollutant_name}: {len(X_val_clean)} (filtered from {len(X_val)})")
    
    if len(X_train_clean) == 0:
        raise ValueError(f"No finite training samples available for {pollutant_name}")
    if len(X_val_clean) == 0:
        raise ValueError(f"No finite validation samples available for {pollutant_name}")
    
    print(f"Starting training for {pollutant_name}...")
    
    if use_generator:
        train_gen = SequenceDataGenerator(
            X_train_clean, y_train_clean, batch_size=batch_size, shuffle=True
        )
        val_gen = SequenceDataGenerator(
            X_val_clean, y_val_clean, batch_size=batch_size, shuffle=False
        )
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            X_train_clean,
            y_train_clean,
            validation_data=(X_val_clean, y_val_clean),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
    
    print(f"Training completed for {pollutant_name}")
    
    gc.collect()
    
    return model, history