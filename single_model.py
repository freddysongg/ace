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
        choices=["mlr", "cnn_lstm"],
        default="mlr",
        help="Model to train and evaluate. (default: mlr)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=("data/input_with_geo_and_interactions_v2.npy"),
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

        target_names = [
            "ozone",
            "pm25_concentration",
            "no2_concentration",
        ]

        target_indices = [column_names.index(name) for name in target_names]

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
        processed_data = data_loader.preprocess_data(
            train_data,
            val_data,
            test_data,
            feature_columns=feature_indices,
            target_columns=target_indices,
            log_transform_targets=[0, 2],  # Ozone & NO2
        )

        sequences_detected = False

        if args.model == "cnn_lstm" and args.sequence_dir:
            seq_dir = Path(args.sequence_dir)
            if not seq_dir.exists():
                raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

            print(f"Sequence directory detected: {seq_dir}. Using pre-built lookback sequences.")
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

                y_true_sample = y_true_orig[idx]
                y_pred_sample = y_pred_orig[idx]

                pollutant_dir = results_dir / "mlr" / pollutant.replace(".", "")
                pollutant_dir.mkdir(parents=True, exist_ok=True)

                evaluate.density_scatter_plot(
                    y_true_sample,
                    y_pred_sample,
                    pollutant,
                    save_path=str(pollutant_dir / f"density_scatter_{pollutant}.png"),
                )

                evaluate.residuals_plot(
                    y_true_sample,
                    y_pred_sample,
                    pollutant,
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

            def _reshape_for_cnn_lstm(x: np.ndarray) -> np.ndarray:
                if x.ndim == 2:
                    return x[:, :, np.newaxis]
                return x

            X_train_cnn = _reshape_for_cnn_lstm(processed_data["X_train"])
            X_val_cnn = _reshape_for_cnn_lstm(processed_data["X_val"])
            X_test_cnn = _reshape_for_cnn_lstm(processed_data["X_test"])

            model, history = train.train_cnn_lstm_model(
                X_train_cnn,
                processed_data["y_train"],
                X_val_cnn,
                processed_data["y_val"],
                epochs=20,
                batch_size=32,
                resume=not args.no_resume,
            )

            cnn_dir = results_dir / "cnn_lstm"
            cnn_dir.mkdir(parents=True, exist_ok=True)

            evaluate.training_history_plot(
                history,
                save_path=str(cnn_dir / "training_history.png"),
                show=False,
            )

            print("Training history plot saved.")

            print(
                "\n==================== CNN+LSTM Evaluation (Test Set) ===================="
            )
            y_test_scaled = processed_data["y_test"]
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

                evaluate.spatial_error_maps_multi(
                    lons=processed_data["lons_test"],
                    lats=processed_data["lats_test"],
                    errors=errors_matrix,
                    pollutant_names=pollutant_names,
                    save_dir=str(spatial_dir),
                )

                print(f"Spatial error maps saved to '{spatial_dir}'.")

            model.save(cnn_dir / "cnn_lstm_model.keras")

            def _py(val):
                import numpy as _np

                if isinstance(val, (_np.floating, _np.integer)):
                    return val.item()
                return val

            hist_serializable = {
                k: [_py(v) for v in vals] for k, vals in history.history.items()
            }

            with open(cnn_dir / "training_history.json", "w") as f:
                json.dump(hist_serializable, f, indent=2)

            print(
                "CNN+LSTM training, evaluation, and artifact saving completed. Results saved to 'results/cnn_lstm/'."
            )
        else:
            print(
                f"Unknown model selection '{args.model}'. Please choose 'mlr' or 'cnn_lstm'."
            )


if __name__ == "__main__":
    main()
