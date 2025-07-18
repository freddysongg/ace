import argparse
from pathlib import Path
import json
import sys

import numpy as np
from . import data_loader, evaluate


_COLUMN_JSON_PATH = Path("data") / "final_column_names.json"

if _COLUMN_JSON_PATH.exists():
    with open(_COLUMN_JSON_PATH, "r") as _f:
        _COLUMN_NAMES = json.load(_f)
else:
    _COLUMN_NAMES = data_loader.get_field_names() or []

_TARGET_NAMES = [
    "ozone",
    "pm25_concentration",
    "no2_concentration",
]

_TARGET_IDX = [_COLUMN_NAMES.index(n) for n in _TARGET_NAMES]
_FEATURE_IDX = [i for i, name in enumerate(_COLUMN_NAMES) if name not in _TARGET_NAMES]

try:
    import pickle
    from tensorflow.keras.models import load_model  # type: ignore
except ImportError as exc:  # pragma: no cover
    print(
        "Required libraries not found – please ensure scikit-learn and TensorFlow are installed."
    )
    raise exc


POLLUTANT_NAMES = ["Ozone", "PM2.5", "NO2"]


def _load_mlr_predictions(proc_data: dict) -> np.ndarray:
    """Load per-pollutant MLR pickle files and return scaled predictions."""
    preds = []
    for pol in POLLUTANT_NAMES:
        model_path = Path("results/mlr") / pol.replace(".", "") / f"mlr_model_{pol}.pkl"
        if not model_path.exists():
            sys.exit(
                f"Missing trained MLR model: {model_path}. Run main.py --model mlr first."
            )
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        preds.append(model.predict(proc_data["X_test"]))
    return np.column_stack(preds)


def _load_cnn_lstm_predictions(proc_data: dict) -> np.ndarray:
    """Load the saved CNN+LSTM Keras model and return scaled predictions."""
    model_path = Path("results/cnn_lstm/cnn_lstm_model.keras")
    if not model_path.exists():
        sys.exit(
            f"Missing CNN+LSTM model at {model_path}. Run main.py --model cnn_lstm first."
        )

    model = load_model(model_path)
    X_test = proc_data["X_test"]
    if X_test.ndim == 2:
        X_test = X_test[:, :, np.newaxis]
    preds = model.predict(X_test, verbose=0)
    if preds.shape != proc_data["y_test"].shape:
        preds = preds.reshape(proc_data["y_test"].shape)
    return preds


def generate_maps(
    model_type: str, data_path: str, year_col: int, save_dir: str
) -> None:
    # 1. Load + preprocess data (includes lat/lon extraction)
    raw = data_loader.load_data(data_path)
    train, val, test = data_loader.chronological_split(raw, year_column_index=year_col)
    proc = data_loader.preprocess_data(
        train,
        val,
        test,
        feature_columns=_FEATURE_IDX,
        target_columns=_TARGET_IDX,
        log_transform_targets=[1, 2],  # PM2.5 & NO2
    )

    if "lons_test" not in proc or "lats_test" not in proc:
        sys.exit(
            "Latitude/longitude columns not found in dataset or FIELD_NAMES – cannot generate maps."
        )

    # 2. Predictions (scaled)
    if model_type == "mlr":
        y_pred_scaled = _load_mlr_predictions(proc)
    elif model_type == "cnn_lstm":
        y_pred_scaled = _load_cnn_lstm_predictions(proc)
    else:
        sys.exit("model_type must be 'mlr' or 'cnn_lstm'.")

    # 3. Convert to original scale & compute errors
    scaler = proc["target_scaler"]
    y_pred_orig = scaler.inverse_transform(y_pred_scaled)
    y_true_orig = scaler.inverse_transform(proc["y_test"])

    errors = y_pred_orig - y_true_orig

    # 4. Save spatial error maps
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    evaluate.spatial_error_maps_multi(
        lons=proc["lons_test"],
        lats=proc["lats_test"],
        errors=errors,
        pollutant_names=POLLUTANT_NAMES,
        save_dir=save_dir,
    )
    print(f"Spatial error maps written to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate spatial error maps without retraining."
    )
    parser.add_argument(
        "--model",
        choices=["mlr", "cnn_lstm"],
        required=True,
        help="Which trained model to use.",
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the .npy dataset."
    )
    parser.add_argument(
        "--year-column",
        type=int,
        default=2,
        help="Index of the year column in the matrix.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/spatial_maps/manual",
        help="Output directory for PNGs.",
    )

    args = parser.parse_args()
    generate_maps(args.model, args.data, args.year_column, args.out)
