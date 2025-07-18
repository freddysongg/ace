import numpy as np
from pathlib import Path

from tensorflow.keras.models import load_model

from src import evaluate, data_loader
from src.hyperparameter_tuning import tune_mlp_hyperparameters

RAW_FILE = "data/input_with_geo_and_interactions_v3.npy"  

raw = data_loader.load_data(RAW_FILE)

train_data, val_data, test_data = data_loader.chronological_split(
    raw,
    train_start=2009,
    train_end=2012,
    val_year=2013,
    test_start=2014,
    test_end=2014,
)

feature_cols = list(range(0, train_data.shape[1] - 3))  

proc = data_loader.preprocess_data(
    train_data,
    val_data,
    test_data,
    feature_columns=feature_cols,
    # log_transform_targets=[0, 2],  
)

# high = proc["y_train_raw"][:, 0] > 10
# print("high-ozone fraction  train:", high.mean())
# high = proc["y_val_raw"][:, 0] > 10
# print("high-ozone fraction  val  :", high.mean())

# print("Robust-scaled y_train stats:",
#       np.percentile(proc["y_train"], [0, 25, 50, 75, 100], axis=0))
# print("Robust-scaled y_val   stats:",
#       np.percentile(proc["y_val"],   [0, 25, 50, 75, 100], axis=0))

for name, raw in [("train", proc["y_train_raw"]),
                  ("val",   proc["y_val_raw"])]:
    print(f"{name} max:", raw.max(axis=0), "median:", np.median(raw, axis=0))

X_train, y_train = proc["X_train"], proc["y_train"]
X_val, y_val     = proc["X_val"], proc["y_val"]

finite_mask_train = np.all(np.isfinite(X_train), axis=1) & np.all(np.isfinite(y_train), axis=1)
finite_mask_val   = np.all(np.isfinite(X_val), axis=1)   & np.all(np.isfinite(y_val), axis=1)

if not finite_mask_train.all():
    n_removed = (~finite_mask_train).sum()
    print(f"Filtering out {n_removed} non-finite training rows …")
    X_train, y_train = X_train[finite_mask_train], y_train[finite_mask_train]

if not finite_mask_val.all():
    n_removed = (~finite_mask_val).sum()
    print(f"Filtering out {n_removed} non-finite validation rows …")
    X_val, y_val = X_val[finite_mask_val], y_val[finite_mask_val]

MAX_TUNE_SAMPLES = 1_000_000  
if X_train.shape[0] > MAX_TUNE_SAMPLES:
    rng = np.random.default_rng(42)
    idx = rng.choice(X_train.shape[0], size=MAX_TUNE_SAMPLES, replace=False)
    X_train, y_train = X_train[idx], y_train[idx]
    print(f"Subsampled training set to {X_train.shape[0]} rows for tuning.")

best_hps, best_model, tuner = tune_mlp_hyperparameters(
    X_train, y_train,
    X_val, y_val,
    max_trials=20,           
    executions_per_trial=2,  
    directory="results/tuner",
    project_name="mlp",
    overwrite=True,  
)

print("Best hyper-parameters:", best_hps.values)



X_test_scaled = proc["X_test"]
y_test_scaled = proc["y_test"]

y_pred_scaled = best_model.predict(X_test_scaled, verbose=0)
if y_pred_scaled.shape != y_test_scaled.shape:
    y_pred_scaled = y_pred_scaled.reshape(y_test_scaled.shape)

target_scaler = proc["target_scaler"]
log_transform_targets = proc.get("log_transform_targets", []) or []

y_test_log = target_scaler.inverse_transform(y_test_scaled)
y_pred_log = target_scaler.inverse_transform(y_pred_scaled)

y_test_orig = np.copy(y_test_log)
y_pred_orig = np.copy(y_pred_log)

for tgt_idx in log_transform_targets:
    y_test_orig[:, tgt_idx] = np.expm1(y_test_orig[:, tgt_idx])
    y_pred_orig[:, tgt_idx] = np.expm1(y_pred_orig[:, tgt_idx])

results_dir = Path("results/mlp")
results_dir.mkdir(parents=True, exist_ok=True)

pollutant_names = ["Ozone", "PM2.5", "NO2"]

evaluate.density_scatter_plots_multi(
    y_test_orig,
    y_pred_orig,
    pollutant_names=pollutant_names,
    save_dir=str(results_dir / "density_scatter"),
)

evaluate.prediction_error_histograms_multi(
    y_test_orig,
    y_pred_orig,
    pollutant_names=pollutant_names,
    save_dir=str(results_dir / "error_histograms"),
    bins=50,
)

evaluate.pred_vs_actual_time_series_slice(
    y_test_orig,
    y_pred_orig,
    pollutant_names=pollutant_names,
    slice_length=500,
    save_path=str(results_dir / "time_series_slice.png"),
)

if "lons_test" in proc and "lats_test" in proc:
    errors_matrix = y_pred_orig - y_test_orig
    spatial_dir = Path("results/spatial_maps/mlp")
    spatial_dir.mkdir(parents=True, exist_ok=True)
    evaluate.spatial_error_maps_multi(
        lons=proc["lons_test"],
        lats=proc["lats_test"],
        errors=errors_matrix,
        pollutant_names=pollutant_names,
        save_dir=str(spatial_dir),
    )

best_model.save(results_dir / "mlp_model_tuned.keras", overwrite=True)

# try:
#     baseline_model = load_model("results/mlp/mlp_model.keras")
#     y_pred_old_scaled = baseline_model.predict(X_test_scaled, verbose=0)
#     if y_pred_old_scaled.shape != y_test_scaled.shape:
#         y_pred_old_scaled = y_pred_old_scaled.reshape(y_test_scaled.shape)
#     y_pred_old = target_scaler.inverse_transform(y_pred_old_scaled)
#     for tgt_idx in log_transform_targets:
#         y_pred_old[:, tgt_idx] = np.expm1(y_pred_old[:, tgt_idx])

#     from sklearn.metrics import r2_score, mean_squared_error

#     rmse_new = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig, multioutput="raw_values"))
#     r2_new   = r2_score(y_test_orig, y_pred_orig,    multioutput="raw_values")
#     rmse_old = np.sqrt(mean_squared_error(y_test_orig, y_pred_old, multioutput="raw_values"))
#     r2_old   = r2_score(y_test_orig, y_pred_old,    multioutput="raw_values")

#     print("Tuned MLP (test) – RMSE:", rmse_new, " R²:", r2_new)
#     print("Baseline MLP (test) – RMSE:", rmse_old, " R²:", r2_old)
# except (OSError, IOError):
#     print("No baseline model found at results/mlp/mlp_model.keras – skipping test-set comparison.")

print("All artifacts saved to 'results/mlp/'.")