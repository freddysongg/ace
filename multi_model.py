import argparse
from pathlib import Path
import os
import json
import pickle
import numpy as np

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
            "If set, CNN+LSTM training will NOT attempt to resume from an existing checkpoint. "
            "Training will start from scratch instead. Has no effect for the MLR model."
        ),
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=("Skip training and evaluate a saved model artifact instead."),
    )

    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help=(
            "Path to a saved model file to load when --eval-only is specified. "
            "If not provided, a sensible default inside the results directory is assumed "
            "based on the selected --model option."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for the script."""
    args = parse_args()

    # Reproducibility – set all relevant PRNG seeds once at the very start
    set_global_seed(42)

    seq_dir_provided = bool(args.sequence_dir)
    seq_dir_exists = False
    if seq_dir_provided:
        _sdir = Path(args.sequence_dir)
        seq_dir_exists = _sdir.exists()
        print(
            f"Sequence directory parameter provided: {_sdir}. Exists on disk: {seq_dir_exists}."
        )
        if seq_dir_exists:
            sequence_dir_to_log = str(_sdir)
        else:
            sequence_dir_to_log = None
    sequences_detected_flag = seq_dir_provided and seq_dir_exists

    data_path = Path(args.data)

    mlflow.set_experiment("AirPollutantPrediction")

    with mlflow.start_run(run_name=f"{args.model.upper()}_pipeline"):
        mlflow.log_param("sequences_detected", sequences_detected_flag)
        if seq_dir_provided and seq_dir_exists:
            mlflow.log_param("sequence_dir", sequence_dir_to_log)

        mlflow.log_param("model", args.model)
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("year_column", args.year_column)
        mlflow.log_param("seed", 42)

        mlflow.log_param("using_prebuilt_sequences", bool(sequences_detected_flag))

        if sequences_detected_flag:
            try:
                _seq_dir = Path(args.sequence_dir)
                _xt_path = _seq_dir / "X_train.npy"
                if _xt_path.exists():
                    _lookback_dim = np.load(_xt_path, mmap_mode="r").shape[1]
                    mlflow.log_param("sequence_lookback", int(_lookback_dim))
            except Exception as _e:  # noqa: BLE001
                import warnings as _warnings

                _warnings.warn(
                    f"Could not infer look-back length from sequence files: {_e}",
                    RuntimeWarning,
                )

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

        lower_cols = [c.lower() for c in column_names]
        for _canon, aliases in canonical_targets:
            idx = next(
                (
                    i
                    for i, col in enumerate(lower_cols)
                    if col in [a.lower() for a in aliases]
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

        pollutant_configs = {
            "Ozone": {
                "target_name": "ozone",
                # "feature_names": feature_names_master,
                "feature_names": [
                    "pollutant_pc2",
                    "month_sin",
                    "month_cos",
                    "so2_x_rsds",
                    "population",
                    "elevation",
                    "elv_x_lat",
                    "road_dense_x_LCZ13",
                    "lon",
                    "rel_humid_range_monmean",
                    "distance_to_coast_km",
                    "tasmax_minus_tasmin_monmean",
                    "oc_x_tasmax",
                    "tasmax_x_pr",
                    "nox_x_rhmin",
                    "bc_x_windspeed",
                    "LCZ_11",
                    "LCZ_14",
                    "LCZ_6",
                    "pollutant_pc3",
                    "LCZ_17",
                    "LCZ_dense_urban",
                ],
                "log_transform": True,
            },
            "PM2.5": {
                "target_name": "pm25_concentration",
                # "feature_names": feature_names_master,
                "feature_names": [
                    "population",
                    "oc_x_tasmax",
                    "bc_x_windspeed",
                    "nox_x_rhmin",
                    "road_dense_x_LCZ13",
                    "so2_x_rsds",
                    "elevation",  # Must-have based on maps
                    "distance_to_coast_km",
                    "LCZ_13",
                    "elv_x_lat",
                    "lat",
                    "LCZ_6",
                    "month_cos",
                    "tasmax_minus_tasmin_monmean",
                    "windspeed_monmean",
                    "LCZ_15",
                    "pollutant_pc2",
                    "rel_humid_range_monmean",
                    "rel_humid_mean_monmean",
                    "month_sin",
                ],
                "log_transform": False,
            },
            "NO2": {
                "target_name": "no2_concentration",
                # "feature_names": feature_names_master,
                "feature_names": [
                    "elevation",
                    "population",
                    "oc_x_tasmax",
                    "bc_x_windspeed",
                    "distance_to_coast_km",
                    "tasmax_minus_tasmin_monmean",
                    "nox_x_rhmin",
                    "road_dense_x_LCZ13",
                    "so2_x_rsds",
                    "elv_x_lat",
                    "LCZ_14",
                    "LCZ_15",
                    "pollutant_pc2",
                    "LCZ_11",
                    "lat",
                    "rel_humid_range_monmean",
                    "windspeed_monmean",
                    "LCZ_13",
                    "LCZ_16",
                    "month_cos",
                ],
                "log_transform": True,
            },
        }

        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        for pollutant, cfg in pollutant_configs.items():
            print(f"\n{'='*10} Building expert model for {pollutant} {'='*10}")

            target_idx = column_names.index(cfg["target_name"])

            feature_indices = [
                column_names.index(f) for f in cfg["feature_names"] if f in column_names
            ]

            if not feature_indices:
                raise ValueError(
                    f"No matching feature columns found in dataset for {pollutant}."
                )
            # Ensure feature_names list exactly matches indices length (for plotting)
            feature_names = [column_names[idx] for idx in feature_indices]

            processed_data = data_loader.preprocess_data(
                train_data,
                val_data,
                test_data,
                feature_columns=feature_indices,
                target_columns=[target_idx],
                log_transform_targets=[0] if cfg["log_transform"] else [],
            )

            tgt_scaler = processed_data.get("target_scaler")
            if "y_train_raw" not in processed_data and tgt_scaler is not None:
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

            if "y_train_raw" not in processed_data:
                processed_data["y_train_raw"] = processed_data["y_train"]

            pollutant_names = [pollutant]

            # ---------------- Raw target visualisations ----------------
            raw_vis_dir = results_dir / "raw_distributions"
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
                save_path=str(raw_vis_dir / f"time_series_slice_{pollutant}.png"),
            )

            if args.model == "mlr":
                mlr_results = train.train_mlr_models(
                    processed_data["X_train"],
                    processed_data["y_train"],
                    processed_data["X_val"],
                    processed_data["y_val"],
                    pollutant_names=pollutant_names,
                )

                model_info = mlr_results[pollutant]
                model = model_info["model"]

                # use the filtered feature_names matching actual model coefficients
                feature_names = feature_names
                X_test = processed_data["X_test"]

                target_scaler = processed_data["target_scaler"]
                log_transform_targets = (
                    processed_data.get("log_transform_targets", []) or []
                )

                y_test_scaled = processed_data["y_test"]
                y_pred_scaled = model.predict(X_test)

                mu = target_scaler.mean_[0]
                sigma = target_scaler.scale_[0]
                y_true_orig = y_test_scaled[:, 0] * sigma + mu
                y_pred_orig = y_pred_scaled * sigma + mu

                if 0 in log_transform_targets:
                    y_true_orig = np.expm1(y_true_orig)
                    y_pred_orig = np.expm1(y_pred_orig)

                n_samples = y_true_orig.shape[0]
                idx = np.random.choice(n_samples, min(5000, n_samples), replace=False)
                if 0 in log_transform_targets:
                    y_true_plot = np.log1p(y_true_orig)
                    y_pred_plot = np.log1p(y_pred_orig)
                else:
                    y_true_plot = y_true_orig
                    y_pred_plot = y_pred_orig

                y_true_sample = y_true_plot[idx]
                y_pred_sample = y_pred_plot[idx]
                label_name = pollutant + (
                    " (log1p)" if 0 in log_transform_targets else ""
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

                evaluate.feature_importance_bar_chart(
                    model,
                    feature_names,
                    pollutant,
                    save_path=str(
                        pollutant_dir / f"feature_importance_{pollutant}.png"
                    ),
                )

                # Persist model & metrics
                with open(pollutant_dir / f"mlr_model_{pollutant}.pkl", "wb") as f:
                    pickle.dump(model, f)

                def _py(val):
                    import numpy as _np

                    return (
                        val.item()
                        if isinstance(val, (_np.floating, _np.integer))
                        else val
                    )

                metrics_only = {
                    k: _py(v) for k, v in model_info.items() if k != "model"
                }
                with open(pollutant_dir / "metrics.json", "w") as f:
                    json.dump(metrics_only, f, indent=2)

                if "lons_test" in processed_data and "lats_test" in processed_data:
                    errors = y_pred_orig - y_true_orig
                    spatial_dir = results_dir / "spatial_maps" / "mlr"
                    spatial_dir.mkdir(parents=True, exist_ok=True)
                    evaluate.spatial_error_map(
                        processed_data["lons_test"],
                        processed_data["lats_test"],
                        errors,
                        pollutant,
                        save_path=str(
                            spatial_dir / f"{pollutant.lower()}_spatial_error_map.png"
                        ),
                    )

            elif args.model == "cnn_lstm":

                lookback = 7
                step = 1

                X_train_cnn, y_train_seq = data_loader.create_lookback_sequences(
                    processed_data["X_train"],
                    processed_data["y_train"],
                    lookback=lookback,
                    step=step,
                )
                X_val_cnn, y_val_seq = data_loader.create_lookback_sequences(
                    processed_data["X_val"],
                    processed_data["y_val"],
                    lookback=lookback,
                    step=step,
                )
                X_test_cnn, y_test_seq = data_loader.create_lookback_sequences(
                    processed_data["X_test"],
                    processed_data["y_test"],
                    lookback=lookback,
                    step=step,
                )

                model, history = train.train_cnn_lstm_model(
                    X_train_cnn,
                    y_train_seq,
                    X_val_cnn,
                    y_val_seq,
                    epochs=20,
                    batch_size=32,
                    resume=not args.no_resume,
                )

                cnn_dir = results_dir / "cnn_lstm" / pollutant.replace(".", "")
                cnn_dir.mkdir(parents=True, exist_ok=True)

                evaluate.training_history_plot(
                    history,
                    save_path=str(cnn_dir / "training_history.png"),
                    show=False,
                )

                y_test_scaled = y_test_seq
                y_pred_scaled = model.predict(X_test_cnn).reshape(y_test_scaled.shape)

                target_scaler = processed_data["target_scaler"]
                log_transform_targets = (
                    processed_data.get("log_transform_targets", []) or []
                )

                y_test_log = target_scaler.inverse_transform(y_test_scaled)
                y_pred_log = target_scaler.inverse_transform(y_pred_scaled)

                y_test_orig = np.copy(y_test_log)
                y_pred_orig = np.copy(y_pred_log)

                if 0 in log_transform_targets:
                    y_test_orig[:, 0] = np.expm1(y_test_orig[:, 0])
                    y_pred_orig[:, 0] = np.expm1(y_pred_orig[:, 0])

                evaluate.density_scatter_plots_multi(
                    y_test_orig,
                    y_pred_orig,
                    pollutant_names=pollutant_names,
                    save_dir=str(cnn_dir / "density_scatter"),
                )

                evaluate.prediction_error_histograms_multi(
                    y_test_orig,
                    y_pred_orig,
                    pollutant_names=pollutant_names,
                    save_dir=str(cnn_dir / "error_histograms"),
                )

                evaluate.pred_vs_actual_time_series_slice(
                    y_test_orig,
                    y_pred_orig,
                    pollutant_names=pollutant_names,
                    slice_length=500,
                    save_path=str(cnn_dir / "time_series_slice.png"),
                )

                if "lons_test" in processed_data and "lats_test" in processed_data:
                    lons_seq = processed_data["lons_test"][lookback:]
                    lats_seq = processed_data["lats_test"][lookback:]
                    errors = (y_pred_orig - y_test_orig).ravel()
                    spatial_dir = results_dir / "spatial_maps" / "cnn_lstm"
                    spatial_dir.mkdir(parents=True, exist_ok=True)
                    evaluate.spatial_error_map(
                        lons=lons_seq,
                        lats=lats_seq,
                        errors=errors,
                        pollutant_name=pollutant,
                        save_path=str(
                            spatial_dir / f"{pollutant.lower()}_spatial_error_map.png"
                        ),
                    )

                model.save(cnn_dir / f"cnn_lstm_model_{pollutant}.keras")

                def _py(val):
                    import numpy as _np

                    return (
                        val.item()
                        if isinstance(val, (_np.floating, _np.integer))
                        else val
                    )

                hist_serializable = {
                    k: [_py(v) for v in vals] for k, vals in history.history.items()
                }
                with open(cnn_dir / "training_history.json", "w") as f:
                    json.dump(hist_serializable, f, indent=2)

            elif args.model == "lstm":
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

                from src import models  # local import to avoid circular

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

                model.save(lstm_dir / "lstm_model.keras")

                hist_serializable = {
                    k: [float(vv) for vv in vals] for k, vals in history.history.items()
                }
                with open(lstm_dir / "training_history.json", "w") as f:
                    json.dump(hist_serializable, f, indent=2)

            elif args.model == "mlp":
                X_train_mlp = processed_data["X_train"]
                X_val_mlp = processed_data["X_val"]
                X_test_mlp = processed_data["X_test"]

                if args.eval_only:
                    import tensorflow as tf  # lazy import

                    default_path = Path("results") / "mlp" / "mlp_model.keras"
                    model_path = (
                        Path(args.model_file) if args.model_file else default_path
                    )
                    if not model_path.exists():
                        raise FileNotFoundError(
                            f"Saved model not found: {model_path.resolve()}. "
                            "Provide --model-file or run training first."
                        )
                    print(f"Loading saved MLP model from {model_path} …")
                    model = tf.keras.models.load_model(model_path, compile=False)
                    history = {"history": {}}
                else:
                    model, history = train.train_mlp_model(
                        X_train_mlp,
                        processed_data["y_train"],
                        X_val_mlp,
                        processed_data["y_val"],
                        epochs=50,
                        batch_size=64,
                    )

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

                y_test_scaled = processed_data["y_test"]
                y_pred_scaled = model.predict(X_test_mlp)

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

                y_val_scaled = processed_data["y_val"]
                y_pred_val_scaled = model.predict(X_val_mlp)
                if y_pred_val_scaled.shape != y_val_scaled.shape:
                    y_pred_val_scaled = y_pred_val_scaled.reshape(y_val_scaled.shape)

                y_val_log = target_scaler.inverse_transform(y_val_scaled)
                y_pred_val_log = target_scaler.inverse_transform(y_pred_val_scaled)

                y_val_orig = np.copy(y_val_log)
                y_pred_val_orig = np.copy(y_pred_val_log)

                for tgt_idx in log_transform_targets:
                    y_test_orig[:, tgt_idx] = np.expm1(y_test_orig[:, tgt_idx])
                    y_pred_orig[:, tgt_idx] = np.expm1(y_pred_orig[:, tgt_idx])
                    y_val_orig[:, tgt_idx] = np.expm1(y_val_orig[:, tgt_idx])
                    y_pred_val_orig[:, tgt_idx] = np.expm1(y_pred_val_orig[:, tgt_idx])

                val_metrics = evaluate.calculate_summary_metrics(
                    y_val_orig, y_pred_val_orig, pollutant_names
                )
                test_metrics = evaluate.calculate_summary_metrics(
                    y_test_orig, y_pred_orig, pollutant_names
                )

                print("\n===== Final Evaluation Metrics =====")
                for pollutant in pollutant_names:
                    print(f"\n--- {pollutant} ---")
                    print(f"  Validation RMSE: {val_metrics[pollutant]['RMSE']:.4f}")
                    print(f"  Validation R²:   {val_metrics[pollutant]['R2']:.4f}")
                    print(f"  Test RMSE:       {test_metrics[pollutant]['RMSE']:.4f}")
                    print(f"  Test R²:         {test_metrics[pollutant]['R2']:.4f}")

                mlflow.log_metrics(
                    {
                        f"test_rmse_{p.lower()}": m["RMSE"]
                        for p, m in test_metrics.items()
                    }
                )
                mlflow.log_metrics(
                    {f"test_r2_{p.lower()}": m["R2"] for p, m in test_metrics.items()}
                )

                for idx, pol in enumerate(pollutant_names):
                    evaluate.plot_keras_evaluation(
                        history,
                        y_test_orig[:, idx],
                        y_pred_orig[:, idx],
                        pollutant_name=pol,
                        save_path=str(
                            mlp_dir / f"evaluation_{pol.lower().replace(' ', '_')}.png"
                        ),
                    )

                evaluate.density_scatter_plots_multi(
                    y_test_orig,
                    y_pred_orig,
                    pollutant_names=pollutant_names,
                    save_dir=str(mlp_dir / "density_scatter"),
                )

                evaluate.prediction_error_histograms_multi(
                    y_test_orig,
                    y_pred_orig,
                    pollutant_names=pollutant_names,
                    save_dir=str(mlp_dir / "error_histograms"),
                )

                evaluate.pred_vs_actual_time_series_slice(
                    y_test_orig,
                    y_pred_orig,
                    pollutant_names=pollutant_names,
                    slice_length=500,
                    save_path=str(mlp_dir / "time_series_slice.png"),
                )

                if "lons_test" in processed_data and "lats_test" in processed_data:
                    errors_matrix = y_pred_orig - y_test_orig
                    spatial_dir = results_dir / "spatial_maps" / "mlp"
                    spatial_dir.mkdir(parents=True, exist_ok=True)

                    evaluate.spatial_error_maps_multi(
                        lons=processed_data["lons_test"],
                        lats=processed_data["lats_test"],
                        errors=errors_matrix,
                        pollutant_names=pollutant_names,
                        save_dir=str(spatial_dir),
                    )

                model.save(mlp_dir / "mlp_model.keras")

                hist_serializable = {
                    k: [float(vv) for vv in vals] for k, vals in history.history.items()
                }
                with open(mlp_dir / "training_history.json", "w") as f:
                    json.dump(hist_serializable, f, indent=2)
            else:
                print(
                    f"Unknown model selection '{args.model}'. Please choose 'mlr', 'cnn_lstm', or 'mlp'."
                )

        return


if __name__ == "__main__":
    main()
