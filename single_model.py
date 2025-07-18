import argparse
from pathlib import Path
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

import mlflow

from src import (
    data_loader,
    train,
    evaluate,
)  # noqa: F401  # pylint: disable=unused-import

from src.utils import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Air pollutant prediction pipeline. Select which model to train "
            "and evaluate."
        )
    )
    parser.add_argument(
        "--model",
        choices=["mlr", "cnn_lstm", "mlp", "lstm"],
        default="mlr",
        help="Model to train and evaluate. (default: mlr)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=("data/input_with_geo_and_interactions_v4.npy"),
        help=(
            "Path to the input .npy data file. Default points to the first "
            "provided dataset in the repository."
        ),
    )

    parser.add_argument(
        "--sequence-dir",
        type=str,
        default=None,
        help=(
            "Directory containing pre-built lookback sequences (X_train.npy, meta.pkl, ...). "
            "Applicable only when --model cnn_lstm. If provided, raw 2-D preprocessing is "
            "still run for bookkeeping but will be replaced by the sequences for training."
        ),
    )
    parser.add_argument(
        "--year-column",
        type=int,
        default=2,
        help="Column index containing the year information. Default 2 for typical datasets.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "If set, CNN+LSTM training will start from scratch instead of resuming from a checkpoint."
        ),
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "If set, skip training and evaluate an already saved model artifact instead."
        ),
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help=(
            "Path to a saved model file to load when --eval-only is specified. "
            "If not provided, a default path inside the results directory is assumed."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for the script."""
    args = parse_args()

    # Reproducibility – set all relevant PRNG seeds once at the very start
    set_global_seed(42)

    data_path = Path(args.data)

    mlflow.set_experiment("AirPollutantPrediction")

    with mlflow.start_run(run_name=f"{args.model.upper()}_pipeline"):
        mlflow.log_param("model", args.model)
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("year_column", args.year_column)
        mlflow.log_param("seed", 42)

        column_names_path = Path("data") / "final_column_names.json"

        if not column_names_path.exists():
            raise FileNotFoundError(
                f"Column names file not found: {column_names_path.resolve()}"
            )

        with open(column_names_path, "r") as f:
            column_names = json.load(f)

        from src.data_loader import _DEFAULT_FIELD_NAMES as DEFAULT_NAMES  # noqa: E402

        if all(
            t.lower() not in [c.lower() for c in column_names]
            for t in [
                "ozone",
                "pm2.5",
                "no2",
                "no2_concentration",
                "ozone",
                "pm25_concentration",
            ]
        ):
            print(
                "Warning: canonical target names not found in saved column list – using default field names."
            )
            column_names = DEFAULT_NAMES

        canonical_targets = [
            ("ozone", ["ozone", "ozone_concentration", "ozone_ppb", "Ozone"]),
            ("pm25", ["pm2.5", "pm25", "pm25_concentration", "PM2.5"]),
            ("no2", ["no2", "no2_concentration", "NO2"]),
        ]

        target_indices = []
        target_names = []

        for _canon, aliases in canonical_targets:
            idx = next(
                (
                    i
                    for i, col in enumerate(column_names)
                    if col.lower() in [a.lower() for a in aliases]
                ),
                None,
            )
            if idx is None:
                raise ValueError(
                    f"Target column for '{_canon}' not found in dataset columns."
                )
            target_indices.append(idx)
            target_names.append(column_names[idx])

        feature_names_master = [
            name for name in column_names if name not in target_names
        ]
        feature_indices = [column_names.index(name) for name in feature_names_master]

        mlflow.log_param("target_names", ",".join(target_names))
        mlflow.log_param("n_features", len(feature_indices))

        if not data_path.exists():
            raise FileNotFoundError(
                f"Provided data file does not exist: {data_path.resolve()}"
            )

        print(f"\nLoading dataset from: {data_path}\n{'-' * 50}")
        raw_data = data_loader.load_data(str(data_path))

        print("\nPerforming chronological split...")
        train_data, val_data, test_data = data_loader.chronological_split(
            raw_data, year_column_index=args.year_column
        )

        print("\nPreprocessing data (features/targets split)...")
        if args.model == "mlp":
            _log_targets = None
            _use_robust = True
        else:
            _log_targets = None
            _use_robust = False

        processed_data = data_loader.preprocess_data(
            train_data,
            val_data,
            test_data,
            feature_columns=feature_indices,
            target_columns=target_indices,
            log_transform_targets=None,
            use_robust_scaler_targets=True,
        )

        sequences_detected = False

        if args.model == "cnn_lstm" and args.sequence_dir:
            seq_dir = Path(args.sequence_dir)
            if not seq_dir.exists():
                raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

            print(
                f"Sequence directory detected: {seq_dir}. Using pre-built lookback sequences."
            )
            sequences_detected = True
            mlflow.log_param("sequences_detected", True)

            import pickle as _pkl

            with open(seq_dir / "meta.pkl", "rb") as _mf:
                meta = _pkl.load(_mf)

            processed_data = {
                "X_train": np.load(seq_dir / "X_train.npy"),
                "y_train": np.load(seq_dir / "y_train.npy"),
                "X_val": np.load(seq_dir / "X_val.npy"),
                "y_val": np.load(seq_dir / "y_val.npy"),
                "X_test": np.load(seq_dir / "X_test.npy"),
                "y_test": np.load(seq_dir / "y_test.npy"),
                "target_scaler": meta.get("target_scaler"),
                "feature_scaler": meta.get("feature_scaler"),
                "log_transform_targets": meta.get("log_transform_targets", []),
                "lons_test": meta.get("lons_test"),
                "lats_test": meta.get("lats_test"),
                "feature_columns": feature_indices,
                "target_columns": target_indices,
            }

            tgt_scaler = processed_data["target_scaler"]
            if tgt_scaler is not None:
                y_train_log = tgt_scaler.inverse_transform(processed_data["y_train"])
                y_val_log = tgt_scaler.inverse_transform(processed_data["y_val"])
                y_test_log = tgt_scaler.inverse_transform(processed_data["y_test"])

                log_tgts = processed_data.get("log_transform_targets", []) or []
                y_train_raw = np.copy(y_train_log)
                y_val_raw = np.copy(y_val_log)
                y_test_raw = np.copy(y_test_log)
                for _idx in log_tgts:
                    y_train_raw[:, _idx] = np.expm1(y_train_raw[:, _idx])
                    y_val_raw[:, _idx] = np.expm1(y_val_raw[:, _idx])
                    y_test_raw[:, _idx] = np.expm1(y_test_raw[:, _idx])

                processed_data["y_train_raw"] = y_train_raw
                processed_data["y_val_raw"] = y_val_raw
                processed_data["y_test_raw"] = y_test_raw
            else:
                processed_data["y_train_raw"] = processed_data["y_train"]

            mlflow.log_param("sequence_dir", str(seq_dir))

        if not sequences_detected:
            mlflow.log_param("sequences_detected", False)

        print("\nData loading and preprocessing complete.")
        print(f"Training features shape: {processed_data['X_train'].shape}")
        print(f"Validation features shape: {processed_data['X_val'].shape}")
        print(f"Test features shape: {processed_data['X_test'].shape}")

        pollutant_names = ["Ozone", "PM2.5", "NO2"]
        print("\nGenerating raw target distribution plots …")
        raw_vis_dir = Path("results") / "raw_distributions"
        raw_vis_dir.mkdir(parents=True, exist_ok=True)

        evaluate.raw_target_histograms(
            processed_data["y_train_raw"],
            pollutant_names=pollutant_names,
            save_dir=str(raw_vis_dir),
        )

        evaluate.target_time_series_slice(
            processed_data["y_train_raw"],
            pollutant_names=pollutant_names,
            slice_length=1000,
            save_path=str(raw_vis_dir / "time_series_slice.png"),
        )

        print("Raw target distribution plots saved to 'results/raw_distributions/'.")

        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        if args.model == "mlr":
            print("\n==================== MLR Training ====================")
            pollutant_names = ["Ozone", "PM2.5", "NO2"]

            mlr_results = train.train_mlr_models(
                processed_data["X_train"],
                processed_data["y_train"],
                processed_data["X_val"],
                processed_data["y_val"],
                pollutant_names=pollutant_names,
            )

            print(
                "\n==================== MLR Evaluation (Test Set) ===================="
            )

            feature_indices = processed_data["feature_columns"]

            if len(column_names) >= max(feature_indices) + 1:
                feature_names = [column_names[idx] for idx in feature_indices]
            else:
                feature_names = [f"feature_{idx}" for idx in feature_indices]

            X_test = processed_data["X_test"]

            target_scaler = processed_data["target_scaler"]
            log_transform_targets = (
                processed_data.get("log_transform_targets", []) or []
            )

            y_test_scaled = processed_data["y_test"]
            y_test_log_space = target_scaler.inverse_transform(y_test_scaled)

            y_test_orig = np.copy(y_test_log_space)
            for tgt_idx in log_transform_targets:
                y_test_orig[:, tgt_idx] = np.expm1(y_test_orig[:, tgt_idx])

            for i, pollutant in enumerate(pollutant_names):
                model_info = mlr_results[pollutant]
                model = model_info["model"]

                y_true_scaled = y_test_scaled[:, i]
                y_pred_scaled = model.predict(X_test)

                mu = target_scaler.mean_[i]
                sigma = target_scaler.scale_[i]
                y_true_orig = y_true_scaled * sigma + mu
                y_pred_orig = y_pred_scaled * sigma + mu

                if i in log_transform_targets:
                    y_true_orig = np.expm1(y_true_orig)
                    y_pred_orig = np.expm1(y_pred_orig)

                n_samples = y_true_orig.shape[0]
                sample_size = min(5000, n_samples)
                idx = np.random.choice(n_samples, sample_size, replace=False)

                if i in log_transform_targets:
                    y_true_plot = np.log1p(y_true_orig)
                    y_pred_plot = np.log1p(y_pred_orig)
                else:
                    y_true_plot = y_true_orig
                    y_pred_plot = y_pred_orig

                y_true_sample = y_true_plot[idx]
                y_pred_sample = y_pred_plot[idx]

                label_name = pollutant + (
                    " (log1p)" if i in log_transform_targets else ""
                )

                pollutant_dir = results_dir / "mlr" / pollutant.replace(".", "")
                pollutant_dir.mkdir(parents=True, exist_ok=True)

                evaluate.density_scatter_plot(
                    y_true_sample,
                    y_pred_sample,
                    label_name,
                    save_path=str(pollutant_dir / f"density_scatter_{pollutant}.png"),
                )

                evaluate.residuals_plot(
                    y_true_sample,
                    y_pred_sample,
                    label_name,
                    save_path=str(pollutant_dir / f"residuals_{pollutant}.png"),
                )

                evaluate.pred_vs_actual_time_series_slice(
                    y_true_orig,
                    y_pred_orig,
                    pollutant_names=[pollutant],
                    slice_length=500,
                    save_path=str(pollutant_dir / f"time_series_{pollutant}.png"),
                )

                evaluate.feature_importance_bar_chart(
                    model,
                    feature_names,
                    pollutant,
                    save_path=str(
                        pollutant_dir / f"feature_importance_{pollutant}.png"
                    ),
                )

                with open(pollutant_dir / f"mlr_model_{pollutant}.pkl", "wb") as f:
                    pickle.dump(model, f)

                def _py(val):
                    import numpy as _np

                    if isinstance(val, (_np.floating, _np.integer)):
                        return val.item()
                    return val

                metrics_only = {
                    k: _py(v) for k, v in model_info.items() if k != "model"
                }
                with open(pollutant_dir / "metrics.json", "w") as f:
                    json.dump(metrics_only, f, indent=2)

                print(f"Generated evaluation plots for {pollutant} at {pollutant_dir}.")

            if "lons_test" in processed_data and "lats_test" in processed_data:
                print("Generating spatial error maps for MLR …")

                preds_scaled_matrix = np.column_stack(
                    [mlr_results[p]["model"].predict(X_test) for p in pollutant_names]
                )

                preds_log_space = target_scaler.inverse_transform(preds_scaled_matrix)
                preds_orig_matrix = np.copy(preds_log_space)

                for tgt_idx in log_transform_targets:
                    preds_orig_matrix[:, tgt_idx] = np.expm1(
                        preds_orig_matrix[:, tgt_idx]
                    )

                errors_matrix = preds_orig_matrix - y_test_orig

                spatial_dir = results_dir / "spatial_maps" / "mlr"
                spatial_dir.mkdir(parents=True, exist_ok=True)

                evaluate.spatial_error_maps_multi(
                    lons=processed_data["lons_test"],
                    lats=processed_data["lats_test"],
                    errors=errors_matrix,
                    pollutant_names=pollutant_names,
                    save_dir=str(spatial_dir),
                )

                print(f"Spatial error maps saved to '{spatial_dir}'.")

            print(
                "\nMLR training, evaluation, and artifact saving completed. Results saved to 'results/mlr/'."
            )

        elif args.model == "cnn_lstm":
            print("\n==================== CNN+LSTM Training ====================")

            pollutant_names = ["Ozone", "PM2.5", "NO2"]

            lookback = 7
            step = 1

            if args.eval_only:
                import tensorflow as tf

                default_path = Path("results") / "cnn_lstm" / "cnn_lstm_model.keras"
                model_path = Path(args.model_file) if args.model_file else default_path
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Saved CNN+LSTM model not found: {model_path.resolve()}. "
                        "Provide --model-file or run training first."
                    )
                print(f"Loading saved CNN+LSTM model from {model_path} …")
                model = tf.keras.models.load_model(model_path, compile=False)
                history = {"history": {}}
            else:
                print("Using memory-efficient data generators for CNN+LSTM training...")

                model, history = train.train_cnn_lstm_model(
                    processed_data["X_train"],
                    processed_data["y_train"],
                    processed_data["X_val"],
                    processed_data["y_val"],
                    epochs=10,
                    batch_size=64,
                    resume=not args.no_resume,
                    use_generator=True,
                )

            X_test_cnn, y_test_seq = data_loader.create_lookback_sequences(
                processed_data["X_test"],
                processed_data["y_test"],
                lookback=lookback,
                step=step,
            )

            cnn_dir = results_dir / "cnn_lstm"
            cnn_dir.mkdir(parents=True, exist_ok=True)

            if history and getattr(history, "history", None):
                evaluate.training_history_plot(
                    history,
                    save_path=str(cnn_dir / "training_history.png"),
                    show=False,
                )
                print("Training history plot saved.")

            print(
                "\n==================== CNN+LSTM Evaluation (Test Set) ===================="
            )
            y_test_scaled = y_test_seq
            y_pred_scaled = model.predict(X_test_cnn)

            if y_pred_scaled.shape != y_test_scaled.shape:
                y_pred_scaled = y_pred_scaled.reshape(y_test_scaled.shape)

            target_scaler = processed_data["target_scaler"]

            log_transform_targets = (
                processed_data.get("log_transform_targets", []) or []
            )

            y_test_log_space = target_scaler.inverse_transform(y_test_scaled)
            y_pred_log_space = target_scaler.inverse_transform(y_pred_scaled)

            y_test_orig = np.copy(y_test_log_space)
            y_pred_orig = np.copy(y_pred_log_space)

            for tgt_idx in log_transform_targets:
                y_test_orig[:, tgt_idx] = np.expm1(y_test_orig[:, tgt_idx])
                y_pred_orig[:, tgt_idx] = np.expm1(y_pred_orig[:, tgt_idx])

            n_samples_total = y_test_orig.shape[0]
            sample_size = min(5000, n_samples_total)
            sample_idx = np.random.choice(n_samples_total, sample_size, replace=False)

            evaluate.density_scatter_plots_multi(
                y_test_orig[sample_idx],
                y_pred_orig[sample_idx],
                pollutant_names=pollutant_names,
                save_dir=str(cnn_dir / "density_scatter"),
            )

            evaluate.prediction_error_histograms_multi(
                y_test_orig,
                y_pred_orig,
                pollutant_names=pollutant_names,
                save_dir=str(cnn_dir / "error_histograms"),
                bins=50,
            )

            evaluate.pred_vs_actual_time_series_slice(
                y_test_orig,
                y_pred_orig,
                pollutant_names=pollutant_names,
                slice_length=500,
                save_path=str(cnn_dir / "time_series_slice.png"),
            )

            if "lons_test" in processed_data and "lats_test" in processed_data:
                print("Generating spatial error maps for CNN+LSTM …")

                errors_matrix = y_pred_orig - y_test_orig

                spatial_dir = results_dir / "spatial_maps" / "cnn_lstm"
                spatial_dir.mkdir(parents=True, exist_ok=True)

                lons_seq = processed_data["lons_test"][lookback:]
                lats_seq = processed_data["lats_test"][lookback:]

                evaluate.spatial_error_maps_multi(
                    lons=lons_seq,
                    lats=lats_seq,
                    errors=errors_matrix,
                    pollutant_names=pollutant_names,
                    save_dir=str(spatial_dir),
                )

                print(f"Spatial error maps saved to '{spatial_dir}'.")

            model.save(cnn_dir / "cnn_lstm_model.keras")

            try:
                import pickle as _pkl

                meta = {
                    "feature_scaler": processed_data.get("feature_scaler"),
                    "target_scaler": processed_data.get("target_scaler"),
                    "log_transform_targets": processed_data.get(
                        "log_transform_targets", []
                    ),
                    "feature_columns": processed_data.get("feature_columns"),
                    "target_columns": processed_data.get("target_columns"),
                }

                with open(cnn_dir / "meta.pkl", "wb") as _mf:
                    _pkl.dump(meta, _mf)
            except Exception as _e:  # noqa: BLE001
                print(f"Warning: failed to save preprocessing metadata – { _e }")

            def _py(val):
                import numpy as _np

                if isinstance(val, (_np.floating, _np.integer)):
                    return val.item()
                return val

            if history and getattr(history, "history", None):
                hist_serializable = {
                    k: [_py(v) for v in vals] for k, vals in history.history.items()
                }
                with open(cnn_dir / "training_history.json", "w") as f:
                    json.dump(hist_serializable, f, indent=2)

            print(
                "CNN+LSTM training, evaluation, and artifact saving completed. Results saved to 'results/cnn_lstm/'."
            )

        elif args.model == "mlp":
            print("\n==================== MLP Training ====================")

            # Per-pollutant configuration system
            pollutant_configs = {
                "Ozone": {
                    "target_index": 0,
                    "use_robust_scaler_targets": False,  # StandardScaler for small range
                    "log_transform": False
                },
                "PM2.5": {
                    "target_index": 1,
                    "use_robust_scaler_targets": True,   # RobustScaler for outliers
                    "log_transform": False
                },
                "NO2": {
                    "target_index": 2,
                    "use_robust_scaler_targets": True,   # RobustScaler for outliers
                    "log_transform": False
                }
            }

            if args.eval_only:
                # -------------------- Load Saved Models --------------------
                import tensorflow as tf  # local import to avoid unnecessary TF load when not needed
                
                # For eval-only mode, we need to load all three per-pollutant models
                pollutant_models = {}
                pollutant_histories = {}
                
                for pollutant_name in pollutant_configs.keys():
                    default_path = Path("results") / "mlp-per-pollutant" / pollutant_name / "mlp_model.keras"
                    model_path = Path(args.model_file) if args.model_file else default_path
                    if not model_path.exists():
                        raise FileNotFoundError(
                            f"Saved model file not found for {pollutant_name}: {model_path.resolve()}. "
                            "Provide --model-file with the correct path or run training first."
                        )
                    print(f"Loading saved {pollutant_name} MLP model from {model_path} …")
                    pollutant_models[pollutant_name] = tf.keras.models.load_model(model_path, compile=False)
                    pollutant_histories[pollutant_name] = {"history": {}}
                
                # For backward compatibility, set model to the first pollutant model
                model = pollutant_models["Ozone"]
                history = pollutant_histories["Ozone"]
            else:
                # -------------------- Train Per-Pollutant Models --------------------
                import gc
                import tensorflow as tf
                
                pollutant_models = {}
                pollutant_histories = {}
                pollutant_processed_data = {}
                
                print("Starting per-pollutant training loop...")
                
                for pollutant_name, config in pollutant_configs.items():
                    print(f"\n==================== Training {pollutant_name} Model ====================")
                    
                    # Preprocess data for this specific pollutant
                    print(f"Preprocessing data for {pollutant_name} with target index {config['target_index']}...")
                    single_pollutant_data = data_loader.preprocess_data(
                        train_data,
                        val_data,
                        test_data,
                        feature_columns=feature_indices,
                        target_columns=target_indices,
                        target_column_index=config['target_index'],  # Single pollutant processing
                        log_transform_targets=None,  # No log transformation
                        use_robust_scaler_targets=config['use_robust_scaler_targets'],
                    )
                    
                    # Store processed data for later evaluation
                    pollutant_processed_data[pollutant_name] = single_pollutant_data
                    
                    print(f"Training data shape for {pollutant_name}: {single_pollutant_data['X_train'].shape}")
                    print(f"Target data shape for {pollutant_name}: {single_pollutant_data['y_train'].shape}")
                    print(f"Using {'RobustScaler' if config['use_robust_scaler_targets'] else 'StandardScaler'} for {pollutant_name}")
                    
                    # Train single-pollutant model
                    model, history = train.train_single_pollutant_mlp_model(
                        single_pollutant_data["X_train"],
                        single_pollutant_data["y_train"].ravel(),  # Convert to 1D for single-pollutant training
                        single_pollutant_data["X_val"],
                        single_pollutant_data["y_val"].ravel(),    # Convert to 1D for single-pollutant training
                        pollutant_name=pollutant_name,
                        epochs=50,
                        batch_size=64,
                    )
                    
                    # Store model and history
                    pollutant_models[pollutant_name] = model
                    pollutant_histories[pollutant_name] = history
                    
                    print(f"Completed training for {pollutant_name}")
                    
                    # Memory cleanup between pollutant training iterations
                    gc.collect()
                    if hasattr(tf.keras.backend, 'clear_session'):
                        tf.keras.backend.clear_session()
                
                print("\nCompleted per-pollutant training loop for all three pollutants.")
                
                # For backward compatibility, set model to the first pollutant model
                model = pollutant_models["Ozone"]
                history = pollutant_histories["Ozone"]

            mlp_dir = results_dir / "mlp"
            mlp_dir.mkdir(parents=True, exist_ok=True)

            if (
                history
                and getattr(history, "history", None)
                and history.history.get("loss")
            ):
                evaluate.training_history_plot(
                    history,
                    save_path=str(mlp_dir / "training_history.png"),
                    show=False,
                )
                print("Training history plot saved.")

            print(
                "\n==================== MLP Evaluation (Test Set) ===================="
            )
            
            # For per-pollutant models, we need to combine predictions from all three models
            if not args.eval_only:
                # Training mode: use per-pollutant processed data and models
                y_pred_val_combined = []
                y_pred_test_combined = []
                y_val_raw_combined = []
                y_test_raw_combined = []
                
                for pollutant_name in ["Ozone", "PM2.5", "NO2"]:
                    single_data = pollutant_processed_data[pollutant_name]
                    single_model = pollutant_models[pollutant_name]
                    
                    # Get predictions for this pollutant
                    y_pred_val_single = single_model.predict(single_data["X_val"])
                    y_pred_test_single = single_model.predict(single_data["X_test"])
                    
                    # Ensure predictions are 1D for single-pollutant models
                    if y_pred_val_single.ndim > 1:
                        y_pred_val_single = y_pred_val_single.ravel()
                    if y_pred_test_single.ndim > 1:
                        y_pred_test_single = y_pred_test_single.ravel()
                    
                    # Transform back to original scale
                    target_scaler = single_data["target_scaler"]
                    y_pred_val_orig = target_scaler.inverse_transform(y_pred_val_single.reshape(-1, 1)).ravel()
                    y_pred_test_orig = target_scaler.inverse_transform(y_pred_test_single.reshape(-1, 1)).ravel()
                    
                    # Get raw targets for this pollutant
                    y_val_raw_single = single_data["y_val_raw"].ravel()
                    y_test_raw_single = single_data["y_test_raw"].ravel()
                    
                    y_pred_val_combined.append(y_pred_val_orig)
                    y_pred_test_combined.append(y_pred_test_orig)
                    y_val_raw_combined.append(y_val_raw_single)
                    y_test_raw_combined.append(y_test_raw_single)
                
                # Stack predictions to create multi-pollutant arrays for evaluation functions
                y_pred_val_orig = np.column_stack(y_pred_val_combined)
                y_pred_test_orig = np.column_stack(y_pred_test_combined)
                y_val_raw = np.column_stack(y_val_raw_combined)
                y_test_raw = np.column_stack(y_test_raw_combined)
                
                # Use the first pollutant's processed data for coordinate information
                processed_data = pollutant_processed_data["Ozone"]
                
            else:
                # Eval-only mode: combine predictions from all three loaded models
                y_pred_val_combined = []
                y_pred_test_combined = []
                y_val_raw_combined = []
                y_test_raw_combined = []
                
                for pollutant_name in ["Ozone", "PM2.5", "NO2"]:
                    config = pollutant_configs[pollutant_name]
                    single_model = pollutant_models[pollutant_name]
                    
                    # Preprocess data for this specific pollutant
                    single_pollutant_data = data_loader.preprocess_data(
                        train_data,
                        val_data,
                        test_data,
                        feature_columns=feature_indices,
                        target_columns=target_indices,
                        target_column_index=config['target_index'],  # Single pollutant processing
                        log_transform_targets=None,  # No log transformation
                        use_robust_scaler_targets=config['use_robust_scaler_targets'],
                    )
                    
                    # Get predictions for this pollutant
                    y_pred_val_single = single_model.predict(single_pollutant_data["X_val"])
                    y_pred_test_single = single_model.predict(single_pollutant_data["X_test"])
                    
                    # Ensure predictions are 1D for single-pollutant models
                    if y_pred_val_single.ndim > 1:
                        y_pred_val_single = y_pred_val_single.ravel()
                    if y_pred_test_single.ndim > 1:
                        y_pred_test_single = y_pred_test_single.ravel()
                    
                    # Transform back to original scale
                    target_scaler = single_pollutant_data["target_scaler"]
                    y_pred_val_orig = target_scaler.inverse_transform(y_pred_val_single.reshape(-1, 1)).ravel()
                    y_pred_test_orig = target_scaler.inverse_transform(y_pred_test_single.reshape(-1, 1)).ravel()
                    
                    # Get raw targets for this pollutant
                    y_val_raw_single = single_pollutant_data["y_val_raw"].ravel()
                    y_test_raw_single = single_pollutant_data["y_test_raw"].ravel()
                    
                    y_pred_val_combined.append(y_pred_val_orig)
                    y_pred_test_combined.append(y_pred_test_orig)
                    y_val_raw_combined.append(y_val_raw_single)
                    y_test_raw_combined.append(y_test_raw_single)
                
                # Stack predictions to create multi-pollutant arrays for evaluation functions
                y_pred_val_orig = np.column_stack(y_pred_val_combined)
                y_pred_test_orig = np.column_stack(y_pred_test_combined)
                y_val_raw = np.column_stack(y_val_raw_combined)
                y_test_raw = np.column_stack(y_test_raw_combined)
                
                # Use the last processed data for coordinate information
                processed_data = single_pollutant_data

            pollutant_names = ["Ozone", "PM2.5", "NO2"]
            val_metrics = evaluate.calculate_summary_metrics(
                y_val_raw, y_pred_val_orig, pollutant_names
            )
            test_metrics = evaluate.calculate_summary_metrics(
                y_test_raw, y_pred_test_orig, pollutant_names
            )

            print("\n===== Final Evaluation Metrics =====")
            for pollutant in pollutant_names:
                print(f"\n--- {pollutant} ---")
                # Raw Metrics
                print(f"  Test R²:         {test_metrics[pollutant]['R2']:.4f}")
                print(f"  Test RMSE:       {test_metrics[pollutant]['RMSE']:.2f}")
                print(f"  Test MAE:        {test_metrics[pollutant]['MAE']:.2f}")
                print(f"  Test Bias:       {test_metrics[pollutant]['Bias']:.2f}")
                
                # --- START: New Normalized Metrics Block ---
                print("  ----------- Normalized -----------")
                nrmse_pct = test_metrics[pollutant].get('NRMSE', float('nan')) * 100
                cv_rmse_pct = test_metrics[pollutant].get('CV_RMSE', float('nan')) * 100
                norm_mae_pct = test_metrics[pollutant].get('Norm_MAE', float('nan')) * 100
                norm_bias_pct = test_metrics[pollutant].get('Norm_Bias', float('nan')) * 100
                
                print(f"  NRMSE (% of Range):   {nrmse_pct:.2f}%")
                print(f"  CV(RMSE) (% of Mean): {cv_rmse_pct:.2f}%")
                print(f"  Norm MAE (% of Mean): {norm_mae_pct:.2f}%")
                print(f"  Norm Bias (% of Mean):{norm_bias_pct:+.2f}%")  # Added '+' to show direction
                # --- END: New Normalized Metrics Block ---

            # Enhanced MLflow logging for per-pollutant models
            print("\n===== MLflow Logging =====")
            
            # Log individual pollutant metrics
            for pollutant in pollutant_names:
                pollutant_key = pollutant.lower().replace(".", "").replace(" ", "_")
                
                # Validation metrics
                mlflow.log_metric(f"val_rmse_{pollutant_key}", val_metrics[pollutant]['RMSE'])
                mlflow.log_metric(f"val_r2_{pollutant_key}", val_metrics[pollutant]['R2'])
                mlflow.log_metric(f"val_mae_{pollutant_key}", val_metrics[pollutant]['MAE'])
                mlflow.log_metric(f"val_bias_{pollutant_key}", val_metrics[pollutant]['Bias'])
                
                # Test metrics
                mlflow.log_metric(f"test_rmse_{pollutant_key}", test_metrics[pollutant]['RMSE'])
                mlflow.log_metric(f"test_r2_{pollutant_key}", test_metrics[pollutant]['R2'])
                mlflow.log_metric(f"test_mae_{pollutant_key}", test_metrics[pollutant]['MAE'])
                mlflow.log_metric(f"test_bias_{pollutant_key}", test_metrics[pollutant]['Bias'])
                
                print(f"Logged metrics for {pollutant}")
            
            # Log aggregate metrics across all pollutants
            avg_val_rmse = np.mean([val_metrics[p]['RMSE'] for p in pollutant_names])
            avg_val_r2 = np.mean([val_metrics[p]['R2'] for p in pollutant_names])
            avg_val_mae = np.mean([val_metrics[p]['MAE'] for p in pollutant_names])
            avg_val_bias = np.mean([val_metrics[p]['Bias'] for p in pollutant_names])
            
            avg_test_rmse = np.mean([test_metrics[p]['RMSE'] for p in pollutant_names])
            avg_test_r2 = np.mean([test_metrics[p]['R2'] for p in pollutant_names])
            avg_test_mae = np.mean([test_metrics[p]['MAE'] for p in pollutant_names])
            avg_test_bias = np.mean([test_metrics[p]['Bias'] for p in pollutant_names])
            
            mlflow.log_metric("avg_val_rmse", avg_val_rmse)
            mlflow.log_metric("avg_val_r2", avg_val_r2)
            mlflow.log_metric("avg_val_mae", avg_val_mae)
            mlflow.log_metric("avg_val_bias", avg_val_bias)
            
            mlflow.log_metric("avg_test_rmse", avg_test_rmse)
            mlflow.log_metric("avg_test_r2", avg_test_r2)
            mlflow.log_metric("avg_test_mae", avg_test_mae)
            mlflow.log_metric("avg_test_bias", avg_test_bias)
            
            print("Logged aggregate metrics across all pollutants")
            
            # Log per-pollutant model configuration
            if not args.eval_only:
                for pollutant_name, config in pollutant_configs.items():
                    pollutant_key = pollutant_name.lower().replace(".", "").replace(" ", "_")
                    scaler_type = "RobustScaler" if config["use_robust_scaler_targets"] else "StandardScaler"
                    mlflow.log_param(f"{pollutant_key}_scaler_type", scaler_type)
                    mlflow.log_param(f"{pollutant_key}_log_transform", config["log_transform"])
                    mlflow.log_param(f"{pollutant_key}_target_index", config["target_index"])
                
                mlflow.log_param("model_type", "per_pollutant_mlp")
                mlflow.log_param("num_separate_models", len(pollutant_configs))
                print("Logged per-pollutant model configuration parameters")
            
            print("MLflow logging completed")
            
            # Generate comprehensive comparison metrics summary
            print("\n===== Generating Comparison Metrics Summary =====")
            
            # Prepare per-pollutant metrics structure for comparison function
            per_pollutant_metrics_summary = {
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics
            }
            
            # Generate and save comparison summary
            comparison_summary_path = results_dir / "mlp-per-pollutant" / "comparison_metrics_summary.json"
            comparison_summary = evaluate.generate_comparison_metrics_summary(
                per_pollutant_metrics=per_pollutant_metrics_summary,
                multi_output_metrics=None,  # Could be added later for comparison with baseline
                save_path=str(comparison_summary_path),
                pollutant_names=pollutant_names
            )
            
            # Log key comparison metrics to MLflow
            mlflow.log_metric("performance_variability_rmse", comparison_summary["comparison_analysis"]["statistical_summary"].get("rmse_coefficient_of_variation", 0.0))
            mlflow.log_metric("performance_range_r2", comparison_summary["comparison_analysis"]["statistical_summary"].get("r2_range", 0.0))
            
            best_rmse_pollutant = comparison_summary["comparison_analysis"]["statistical_summary"].get("best_rmse_pollutant")
            best_r2_pollutant = comparison_summary["comparison_analysis"]["statistical_summary"].get("best_r2_pollutant")
            
            if best_rmse_pollutant:
                mlflow.log_param("best_rmse_pollutant", best_rmse_pollutant)
            if best_r2_pollutant:
                mlflow.log_param("best_r2_pollutant", best_r2_pollutant)
            
            print(f"Comparison metrics summary saved to: {comparison_summary_path}")
            print(f"Best RMSE performance: {best_rmse_pollutant}")
            print(f"Best R² performance: {best_r2_pollutant}")
            print("Comparison metrics summary generation completed")

            has_history = (
                history
                and getattr(history, "history", None)
                and history.history.get("loss")
            )

            # ---------- SHAP & permutation importance ----------
            if has_history:
                # SHAP (optional, can be slow) - use first pollutant model for demonstration
                try:
                    first_model = pollutant_models["Ozone"] if not args.eval_only else model
                    X_train_subset = processed_data["X_train"][:1000]
                    evaluate.shap_summary_plot(
                        first_model,
                        X_train_subset,
                        feature_names_master,
                        save_path=str(mlp_dir / "shap_summary.png"),
                    )
                except Exception as exc:
                    print(f"SHAP summary plot skipped: {exc}")

                # Permutation importance (single target for speed) - use Ozone model
                try:
                    ozone_model = pollutant_models["Ozone"] if not args.eval_only else model
                    # For per-pollutant models, use the single-pollutant target data
                    if not args.eval_only:
                        ozone_data = pollutant_processed_data["Ozone"]
                        y_val_single = ozone_data["y_val"].ravel()
                        X_val_single = ozone_data["X_val"]
                    else:
                        y_val_single = y_val_raw[:, 0]  # Use first column for Ozone
                        X_val_single = processed_data["X_val"]
                    
                    evaluate.permutation_importance_plot(
                        ozone_model,
                        X_val_single,
                        y_val_single,
                        feature_names_master,
                        pollutant_name="Ozone",
                        save_path=str(mlp_dir / "perm_importance_ozone.png"),
                    )
                except Exception as exc:
                    print(f"Permutation importance skipped: {exc}")
            else:
                print("Skipped SHAP & permutation importance (no training history).")

            # ---------- per-pollutant RMSE/R² panel ----------
            if has_history and history.history.get("mse"):
                for i, pollutant in enumerate(pollutant_names):
                    # Use individual pollutant history for per-pollutant models
                    if not args.eval_only:
                        pollutant_history = pollutant_histories[pollutant]
                        if (pollutant_history and 
                            hasattr(pollutant_history, "history") and 
                            pollutant_history.history.get("mse")):
                            evaluate.plot_keras_evaluation(
                                pollutant_history,
                                y_test_raw[:, i],
                                y_pred_test_orig[:, i],
                                pollutant_name=pollutant,
                                save_path=str(
                                    mlp_dir
                                    / f"evaluation_{pollutant.lower().replace(' ', '_')}.png"
                                ),
                            )
                        else:
                            print(f"Skipped plot_keras_evaluation for {pollutant} (no 'mse' in history).")
                    else:
                        print(f"Skipped plot_keras_evaluation for {pollutant} (eval-only mode).")
            else:
                print("Skipped plot_keras_evaluation (no ‘mse’ in history).")

            evaluate.density_scatter_plots_multi(
                y_test_raw,
                y_pred_test_orig,
                pollutant_names=pollutant_names,
                save_dir=str(mlp_dir / "density_scatter"),
            )

            evaluate.prediction_error_histograms_multi(
                y_test_raw,
                y_pred_test_orig,
                pollutant_names=pollutant_names,
                save_dir=str(mlp_dir / "error_histograms"),
                bins=50,
            )

            evaluate.pred_vs_actual_time_series_slice(
                y_test_raw,
                y_pred_test_orig,
                pollutant_names=pollutant_names,
                slice_length=500,
                save_path=str(mlp_dir / "time_series_slice.png"),
            )

            if "lons_test" in processed_data and "lats_test" in processed_data:
                print("Generating spatial error maps for MLP …")

                errors_matrix = y_pred_test_orig - y_test_raw

                # Use per-pollutant directory name to distinguish from multi-output models
                spatial_dir = results_dir / "spatial_maps" / "mlp-per-pollutant"
                spatial_dir.mkdir(parents=True, exist_ok=True)

                evaluate.spatial_error_maps_multi(
                    lons=processed_data["lons_test"],
                    lats=processed_data["lats_test"],
                    errors=errors_matrix,
                    pollutant_names=pollutant_names,
                    save_dir=str(spatial_dir),
                )

                print(f"Spatial error maps saved to '{spatial_dir}'.")

            # Save per-pollutant models and artifacts
            if not args.eval_only:
                # Training mode: save each pollutant model separately
                for pollutant_name in ["Ozone", "PM2.5", "NO2"]:
                    pollutant_dir = results_dir / "mlp-per-pollutant" / pollutant_name
                    pollutant_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Get individual model, history, and data for this pollutant
                    single_model = pollutant_models[pollutant_name]
                    single_history = pollutant_histories[pollutant_name]
                    single_data = pollutant_processed_data[pollutant_name]
                    config = pollutant_configs[pollutant_name]
                    
                    # Save individual model
                    single_model.save(pollutant_dir / "mlp_model.keras")
                    print(f"Saved {pollutant_name} model to {pollutant_dir / 'mlp_model.keras'}")
                    
                    # Calculate individual pollutant metrics
                    pollutant_index = config["target_index"]
                    y_val_single = y_val_raw[:, pollutant_index]
                    y_test_single = y_test_raw[:, pollutant_index]
                    y_pred_val_single = y_pred_val_orig[:, pollutant_index]
                    y_pred_test_single = y_pred_test_orig[:, pollutant_index]
                    
                    # Calculate metrics for this pollutant
                    single_val_metrics = evaluate.calculate_summary_metrics(
                        y_val_single.reshape(-1, 1), 
                        y_pred_val_single.reshape(-1, 1), 
                        [pollutant_name]
                    )
                    single_test_metrics = evaluate.calculate_summary_metrics(
                        y_test_single.reshape(-1, 1), 
                        y_pred_test_single.reshape(-1, 1), 
                        [pollutant_name]
                    )
                    
                    # Save individual metrics
                    metrics_data = {
                        "validation_metrics": single_val_metrics[pollutant_name],
                        "test_metrics": single_test_metrics[pollutant_name],
                        "pollutant_name": pollutant_name,
                        "target_index": config["target_index"],
                        "scaler_type": "RobustScaler" if config["use_robust_scaler_targets"] else "StandardScaler",
                        "log_transform": config["log_transform"]
                    }
                    
                    with open(pollutant_dir / "metrics.json", "w") as f:
                        # Convert to JSON-serializable format before saving
                        serializable_metrics = evaluate.convert_to_json_serializable(metrics_data)
                        json.dump(serializable_metrics, f, indent=2)
                    
                    print(f"Saved {pollutant_name} metrics: Test RMSE={single_test_metrics[pollutant_name]['RMSE']:.4f}, R²={single_test_metrics[pollutant_name]['R2']:.4f}")
                    
                    # Save preprocessing configuration metadata for this pollutant
                    try:
                        import pickle as _pkl
                        
                        meta = {
                            "feature_scaler": single_data.get("feature_scaler"),
                            "target_scaler": single_data.get("target_scaler"),
                            "log_transform_targets": single_data.get("log_transform_targets", []),
                            "feature_columns": single_data.get("feature_columns"),
                            "target_columns": single_data.get("target_columns"),
                            "target_column_index": single_data.get("target_column_index"),
                            "pollutant_name": pollutant_name,
                            "preprocessing_config": {
                                "use_robust_scaler_targets": config["use_robust_scaler_targets"],
                                "log_transform": config["log_transform"],
                                "scaler_type": "RobustScaler" if config["use_robust_scaler_targets"] else "StandardScaler"
                            }
                        }
                        
                        with open(pollutant_dir / "meta.pkl", "wb") as _mf:
                            _pkl.dump(meta, _mf)
                        
                        print(f"Saved {pollutant_name} preprocessing metadata with {meta['preprocessing_config']['scaler_type']}")
                    except Exception as _e:  # noqa: BLE001
                        print(f"Warning: failed to save preprocessing metadata for {pollutant_name} – {_e}")
                    
                    # Save training history for this pollutant
                    if hasattr(single_history, "history"):
                        hist_data = single_history.history
                    else:
                        hist_data = single_history.get("history", {})
                    
                    hist_serializable = {
                        k: [float(vv) for vv in vals] for k, vals in hist_data.items()
                    }
                    
                    with open(pollutant_dir / "training_history.json", "w") as f:
                        json.dump(hist_serializable, f, indent=2)
                    
                    # Generate individual pollutant evaluation plots
                    print(f"Generating evaluation plots for {pollutant_name}...")
                    
                    # Create subdirectories for individual pollutant plots
                    density_dir = pollutant_dir / "density_scatter"
                    density_dir.mkdir(parents=True, exist_ok=True)
                    
                    error_dir = pollutant_dir / "error_histograms"
                    error_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate density scatter plot for this pollutant
                    n_samples_total = y_test_single.shape[0]
                    sample_size = min(5000, n_samples_total)
                    sample_idx = np.random.choice(n_samples_total, sample_size, replace=False)
                    
                    evaluate.density_scatter_plot(
                        y_test_single[sample_idx],
                        y_pred_test_single[sample_idx],
                        pollutant_name,
                        save_path=str(density_dir / f"density_scatter_{pollutant_name.lower().replace('.', '')}.png")
                    )
                    
                    # Generate error histogram for this pollutant
                    errors_single = y_pred_test_single - y_test_single
                    
                    plt.figure(figsize=(8, 6))
                    plt.hist(errors_single, bins=50, alpha=0.7, edgecolor='black')
                    plt.xlabel(f'Prediction Error ({pollutant_name})')
                    plt.ylabel('Frequency')
                    plt.title(f'Prediction Error Distribution - {pollutant_name}')
                    plt.grid(True, alpha=0.3)
                    
                    error_path = error_dir / f"error_histogram_{pollutant_name.lower().replace('.', '')}.png"
                    plt.savefig(error_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Generate time series comparison for this pollutant
                    evaluate.pred_vs_actual_time_series_slice(
                        y_test_single.reshape(-1, 1),
                        y_pred_test_single.reshape(-1, 1),
                        pollutant_names=[pollutant_name],
                        slice_length=500,
                        save_path=str(pollutant_dir / f"time_series_{pollutant_name.lower().replace('.', '')}.png")
                    )
                    
                    print(f"Completed artifact saving for {pollutant_name} in {pollutant_dir}")
                
                print("\nPer-pollutant artifact saving completed:")
                print("├── results/mlp-per-pollutant/")
                for pollutant_name in ["Ozone", "PM2.5", "NO2"]:
                    print(f"│   ├── {pollutant_name}/")
                    print(f"│   │   ├── mlp_model.keras")
                    print(f"│   │   ├── metrics.json")
                    print(f"│   │   ├── training_history.json")
                    print(f"│   │   ├── meta.pkl")
                    print(f"│   │   ├── density_scatter/")
                    print(f"│   │   ├── error_histograms/")
                    print(f"│   │   └── time_series_{pollutant_name.lower().replace('.', '')}.png")
            else:
                # Eval-only mode: maintain backward compatibility by saving the first model
                model.save(mlp_dir / "mlp_model.keras")
                
                try:
                    import pickle as _pkl

                    meta = {
                        "feature_scaler": processed_data.get("feature_scaler"),
                        "target_scaler": processed_data.get("target_scaler"),
                        "log_transform_targets": processed_data.get(
                            "log_transform_targets", []
                        ),
                        "feature_columns": processed_data.get("feature_columns"),
                        "target_columns": processed_data.get("target_columns"),
                    }

                    with open(mlp_dir / "meta.pkl", "wb") as _mf:
                        _pkl.dump(meta, _mf)
                except Exception as _e:  # noqa: BLE001
                    print(f"Warning: failed to save preprocessing metadata – { _e }")

                # Handle both dict and Keras History object types
                if hasattr(history, "history"):
                    # Keras History object
                    hist_data = history.history
                else:
                    # Dictionary (from eval-only mode)
                    hist_data = history.get("history", {})

                hist_serializable = {
                    k: [float(vv) for vv in vals] for k, vals in hist_data.items()
                }

                with open(mlp_dir / "training_history.json", "w") as f:
                    json.dump(hist_serializable, f, indent=2)

            if not args.eval_only:
                print(
                    "Per-pollutant MLP training, evaluation, and artifact saving completed. "
                    "Results saved to 'results/mlp-per-pollutant/' with separate models for each pollutant."
                )
            else:
                print(
                    "MLP evaluation completed. Results saved to 'results/mlp/'."
                )

        elif args.model == "lstm":
            print("\n==================== LSTM Training ====================")

            pollutant_names = ["Ozone", "PM2.5", "NO2"]

            lookback = 7
            step = 1

            X_train_seq, y_train_seq = data_loader.create_lookback_sequences(
                processed_data["X_train"],
                processed_data["y_train"],
                lookback,
                step,
            )
            X_val_seq, y_val_seq = data_loader.create_lookback_sequences(
                processed_data["X_val"],
                processed_data["y_val"],
                lookback,
                step,
            )
            X_test_seq, y_test_seq = data_loader.create_lookback_sequences(
                processed_data["X_test"],
                processed_data["y_test"],
                lookback,
                step,
            )

            from src import models

            model, history = train.train_cnn_lstm_model(
                X_train_seq,
                y_train_seq,
                X_val_seq,
                y_val_seq,
                epochs=30,
                batch_size=64,
                resume=False,
                model_builder=models.get_simple_lstm_model,
            )

            lstm_dir = results_dir / "lstm"
            lstm_dir.mkdir(parents=True, exist_ok=True)

            evaluate.training_history_plot(
                history,
                save_path=str(lstm_dir / "training_history.png"),
                show=False,
            )

            print("Training history plot saved.")

            print(
                "\n==================== LSTM Evaluation (Test Set) ===================="
            )

            y_test_scaled = y_test_seq
            y_pred_scaled = model.predict(X_test_seq)
            if y_pred_scaled.shape != y_test_scaled.shape:
                y_pred_scaled = y_pred_scaled.reshape(y_test_scaled.shape)

            target_scaler = processed_data["target_scaler"]
            log_transform_targets = (
                processed_data.get("log_transform_targets", []) or []
            )

            y_test_log = target_scaler.inverse_transform(y_test_scaled)
            y_pred_log = target_scaler.inverse_transform(y_pred_scaled)

            y_test_orig = np.copy(y_test_log)
            y_pred_orig = np.copy(y_pred_log)

            for tgt_idx in log_transform_targets:
                y_test_orig[:, tgt_idx] = np.expm1(y_test_orig[:, tgt_idx])
                y_pred_orig[:, tgt_idx] = np.expm1(y_pred_orig[:, tgt_idx])

            for idx, pol in enumerate(pollutant_names):
                evaluate.plot_keras_evaluation(
                    history,
                    y_test_orig[:, idx],
                    y_pred_orig[:, idx],
                    pollutant_name=pol,
                    save_path=str(
                        lstm_dir / f"evaluation_{pol.lower().replace(' ', '_')}.png"
                    ),
                )

            evaluate.density_scatter_plots_multi(
                y_test_orig,
                y_pred_orig,
                pollutant_names=pollutant_names,
                save_dir=str(lstm_dir / "density_scatter"),
            )

            evaluate.prediction_error_histograms_multi(
                y_test_orig,
                y_pred_orig,
                pollutant_names=pollutant_names,
                save_dir=str(lstm_dir / "error_histograms"),
            )

            evaluate.pred_vs_actual_time_series_slice(
                y_test_orig,
                y_pred_orig,
                pollutant_names=pollutant_names,
                slice_length=500,
                save_path=str(lstm_dir / "time_series_slice.png"),
            )

            if "lons_test" in processed_data and "lats_test" in processed_data:
                print("Generating spatial error maps for LSTM …")

                lons_seq = processed_data["lons_test"][lookback:]
                lats_seq = processed_data["lats_test"][lookback:]

                errors_matrix = y_pred_orig - y_test_orig

                spatial_dir = results_dir / "spatial_maps" / "lstm"
                spatial_dir.mkdir(parents=True, exist_ok=True)

                evaluate.spatial_error_maps_multi(
                    lons=lons_seq,
                    lats=lats_seq,
                    errors=errors_matrix,
                    pollutant_names=pollutant_names,
                    save_dir=str(spatial_dir),
                )

                print(f"Spatial error maps saved to '{spatial_dir}'.")

            model.save(lstm_dir / "lstm_model.keras")

            hist_serializable = {
                k: [float(vv) for vv in vals] for k, vals in history.history.items()
            }

            with open(lstm_dir / "training_history.json", "w") as f:
                json.dump(hist_serializable, f, indent=2)

            print(
                "LSTM training, evaluation, and artifact saving completed. Results saved to 'results/lstm/'."
            )

        else:
            print(
                f"Unknown model selection '{args.model}'. Please choose 'mlr', 'cnn_lstm', 'mlp', or 'lstm'."
            )


if __name__ == "__main__":
    main()
