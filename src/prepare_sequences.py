import argparse
from pathlib import Path
import json
import numpy as np
import mlflow

from src import data_loader
from src.utils import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare look-back window sequences for CNN+LSTM training."
    )
    p.add_argument(
        "--data",
        required=True,
        type=str,
        help="Path to the *pre-engineered* 2-D .npy file (45 columns).",
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=7,
        help="Number of past timesteps to include (>=1).",
    )
    p.add_argument(
        "--step",
        type=int,
        default=1,
        help="Stride between consecutive sequences (default 1).",
    )
    p.add_argument(
        "--year-column",
        type=int,
        default=2,
        help="Index of the year column in the matrix (default 2).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/sequences",
        help="Directory to write X_train.npy, y_train.npy, etc.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    set_global_seed(42)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    column_json = Path("data") / "final_column_names.json"
    with open(column_json, "r") as f:
        column_names = json.load(f)

    target_names = ["ozone", "pm25_concentration", "no2_concentration"]
    target_idx = [column_names.index(n) for n in target_names]
    feature_idx = [i for i, name in enumerate(column_names) if name not in target_names]

    mlflow.set_experiment("SequencePreparation")
    with mlflow.start_run(run_name="build_sequences"):
        mlflow.log_params(
            {
                "lookback": args.lookback,
                "step": args.step,
                "data_file": args.data,
                "year_column": args.year_column,
                "out_dir": str(out_dir),
            }
        )

        raw = data_loader.load_data(args.data)
        train, val, test = data_loader.chronological_split(
            raw, year_column_index=args.year_column
        )

        proc = data_loader.preprocess_data(
            train,
            val,
            test,
            feature_columns=feature_idx,
            target_columns=target_idx,
            log_transform_targets=[0, 2],  # Ozone & NO2
        )

        X_train, y_train = data_loader.create_lookback_sequences(
            proc["X_train"], proc["y_train"], args.lookback, args.step
        )
        X_val, y_val = data_loader.create_lookback_sequences(
            proc["X_val"], proc["y_val"], args.lookback, args.step
        )
        X_test, y_test = data_loader.create_lookback_sequences(
            proc["X_test"], proc["y_test"], args.lookback, args.step
        )

        np.save(out_dir / "X_train.npy", X_train)
        np.save(out_dir / "y_train.npy", y_train)
        np.save(out_dir / "X_val.npy", X_val)
        np.save(out_dir / "y_val.npy", y_val)
        np.save(out_dir / "X_test.npy", X_test)
        np.save(out_dir / "y_test.npy", y_test)

        import pickle as _pkl

        meta = {
            "target_scaler": proc["target_scaler"],
            "feature_scaler": proc["feature_scaler"],
            "log_transform_targets": proc.get("log_transform_targets", []),
            "lons_test": proc.get("lons_test"),
            "lats_test": proc.get("lats_test"),
        }

        with open(out_dir / "meta.pkl", "wb") as _mf:
            _pkl.dump(meta, _mf)

        shapes = {
            "X_train": X_train.shape,
            "X_val": X_val.shape,
            "X_test": X_test.shape,
            "lookback": args.lookback,
        }

        def _to_serializable(val):
            if isinstance(val, tuple):
                return list(val)
            return val

        print("Saved sequences to", out_dir)
        print(json.dumps({k: _to_serializable(v) for k, v in shapes.items()}, indent=2))

        mlflow.log_metrics(
            {f"n_{k.lower()}": v[0] for k, v in shapes.items() if k.startswith("X_")}
        )
        mlflow.log_artifacts(str(out_dir), artifact_path="sequence_files")


if __name__ == "__main__":
    main()
