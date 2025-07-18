from __future__ import annotations

from typing import Optional, Tuple, Union

import tensorflow as tf
import keras_tuner as kt  # type: ignore

from .models import get_mlp_model


def _build_mlp_model(hp: kt.HyperParameters, input_dim: int, num_outputs: int) -> tf.keras.Model:  # noqa: D401, E501
    """Construct an MLP model parameterised by *hp*.

    The search space is intentionally kept compact but expressive:

    * *hidden_layers* – number of hidden Dense layers (1 to 5)
    * *units_i* – units in the *i*-th hidden layer, 32–256 with step 32
    * *dropout* – shared dropout rate across layers, 0–0.5 (step 0.1)
    * *learning_rate* – log-uniform between 1e-4 and 1e-2
    """
    hidden_layers = hp.Int("hidden_layers", min_value=1, max_value=3, step=1)

    hidden_units = []
    for layer_idx in range(hidden_layers):
        units = hp.Int(
            f"units_{layer_idx}", min_value=32, max_value=256, step=32, default=64
        )
        hidden_units.append(units)

    dropout = hp.Float("dropout", min_value=0.1, max_value=0.6, step=0.1, default=0.4)

    l2_reg = hp.Float("l2", min_value=1e-5, max_value=1e-2, sampling="log", default=1e-4)

    learning_rate = hp.Float(
        "learning_rate", min_value=5e-6, max_value=5e-3, sampling="log", default=5e-5
    )

    model = get_mlp_model(
        input_dim=input_dim,
        num_outputs=num_outputs,
        hidden_units=tuple(hidden_units),
        dropout=dropout,
        l2_reg=l2_reg,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
        ],
    )
    return model


def tune_mlp_hyperparameters(
    X_train: "np.ndarray",  # type: ignore
    y_train: "np.ndarray",  # type: ignore
    X_val: "np.ndarray",  # type: ignore
    y_val: "np.ndarray",  # type: ignore
    *,
    max_trials: int = 20,
    executions_per_trial: int = 1,
    directory: str = "tuner_results",
    project_name: str = "mlp_tuning",
    overwrite: bool = True,
    seed: Optional[int] = None,
) -> Tuple[kt.HyperParameters, tf.keras.Model, kt.Tuner]:
    """Run Bayesian optimisation to find the best MLP hyper-parameters.

    Parameters
    ----------
    X_train, y_train, X_val, y_val
        Numpy arrays holding the train/validation splits.
    max_trials
        Maximum number of *different* hyper-parameter configurations to try.
    executions_per_trial
        How many times to repeat the training for the same configuration to
        reduce stochasticity.
    directory, project_name
        Where to save the tuner artefacts.
    overwrite
        Whether to clear any existing tuner state in *directory/project_name*.
    seed
        Random seed for reproducibility.

    Returns
    -------
    best_hps, best_model, tuner
        *best_hps* – best performing hyper-parameters;
        *best_model* – the corresponding trained Keras model;
        *tuner* – the Keras Tuner instance for inspection.
    """
    import numpy as np

    if X_train.ndim != 2:
        raise ValueError("X_train must be 2-D (samples, features) for MLP tuning.")

    num_outputs: int = y_train.shape[1] if y_train.ndim == 2 else 1

    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: _build_mlp_model(hp, X_train.shape[1], num_outputs),
        objective=kt.Objective("val_mse", direction="min"),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=directory,
        project_name=project_name,
        overwrite=overwrite,
        seed=seed,
    )

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    tuner.search(
        X_train,
        y_train,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[stop_early],
        verbose=1,
    )

    best_hps = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    return best_hps, best_model, tuner 