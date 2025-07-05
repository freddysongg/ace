# Air Pollutant Prediction Models

Multi-model framework for predicting ground-level concentrations of key air pollutants (Ozone, PM2.5, NO2) across the contiguous United States.

---

## Table of Contents
1. Project Overview
2. Directory Structure
3. Installation
4. Data
5. Running the Pipelines
6. Experiment Tracking & Visualisation
7. Testing
8. Results Directory Layout
9. Citation

---

## 1. Project Overview
This project provides two different approaches to air-quality prediction:

1. **Multiple Linear Regression (MLR)** – fast, interpretable baseline.
2. **CNN + LSTM** – deep-learning architecture that captures spatial and temporal dependencies.

The full workflow includes data loading & preprocessing, model training, evaluation, and automatic logging of metrics and artifacts to MLflow.

---

## 2. Directory Structure
```text
├── src/                 # Core source code
│   ├── data_loader.py   # Loading / preprocessing utilities
│   ├── models.py        # Model definitions (MLR, CNN+LSTM)
│   ├── train.py         # Training routines
│   └── evaluate.py      # Evaluation & plotting utilities
├── data/               # Input datasets (.npy) and SEDAC spreadsheet
├── results/             # Generated figures & model artifacts
├── tests/               # Unit tests for key functionality
├── notebooks/           # Exploratory notebooks (archived)
├── requirements.txt     # Python dependencies
├── multi_model.py       # Expert-per-pollutant pipeline (one model per pollutant)
├── single_model.py      # Unified multi-output pipeline (single model for all pollutants)
```

---

## 3. Installation

1. Clone the repository and change into the project directory.
2. (Recommended) Create a virtual environment, e.g. with `python -m venv .venv` and activate it.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

TensorFlow wheels are platform-specific; if you encounter issues, refer to the official installation guide.

---

## 4. Data
All input datasets live in the `data/` directory:

* `data/input_version_1 - ozone_matched_with_all_meteo_features_and_LCZ_and_Pop_and_Emission_weighted_2001-2014.npy`
* `data/input_version_2 - 0p03grid_ozone_population_urbanfrac_road_lu_emissions_meteo_pm25_no2_v2.npy`
* `data/SEDAC Data Sets and Granules 10 April 2025.xlsx` – metadata reference from SEDAC.

If you store files elsewhere, either update the `--data` argument when running `main.py` or modify the default path in the script.

---

## 5. Running the Pipelines

Two high-level entry-points are provided:

| Script | Purpose |
| ------ | ------- |
| `multi_model.py` | Trains **independent / expert** models for each pollutant. Useful when you want tailor-made feature sets and potentially different algorithms per target (e.g. MLR for Ozone, CNN-LSTM for PM2.5). |
| `single_model.py` | Trains a **single multi-output** model that predicts all pollutants simultaneously. |

Both scripts share a common CLI:

```text
--model {mlr,cnn_lstm}   # Choose algorithm (default: mlr)
--data PATH              # Path to .npy dataset (default points to repository file)
--sequence-dir PATH      # Pre-generated look-back sequences (CNN-LSTM only)
--year-column INT        # Index of year column for chronological split (default: 2)
--no-resume              # Start CNN-LSTM training **from scratch** (ignore checkpoints)
```

Example – train a fresh CNN-LSTM model for PM2.5 expert pipeline:

```bash
python multi_model.py --model cnn_lstm --no-resume
```

Example – continue training the unified multi-output CNN-LSTM from last checkpoint:

```bash
python single_model.py --model cnn_lstm
```

During execution the scripts will:

1. Load and chronologically split the dataset.
2. (Optional) Replace raw data with sequences from `--sequence-dir` for CNN-LSTM.
3. Train the chosen model.
4. Evaluate on validation & test sets, generating plots.
5. Log everything (parameters, metrics, artifacts, TensorBoard logs, checkpoints) to `results/` and MLflow.

---

## 6. Experiment Tracking & Visualisation

### MLflow

All runs are logged to the **AirPollutantPrediction** experiment. Launch the UI with:

```bash
mlflow ui
```

Browse `http://localhost:5000` to inspect parameters, metrics, and artefacts.

### TensorBoard

Training curves for CNN-LSTM are also recorded with TensorBoard. Start it via:

```bash
tensorboard --logdir results/logs
```

Checkpointed models are stored under `results/checkpoints/<run_id>/` and are automatically resumed unless `--no-resume` is supplied.

---

## 7. Testing

Execute unit tests with **pytest**:

```bash
pytest -q
```

---

## 8. Results Directory Layout

```text
results/
├── raw_distributions/          # Histograms & time-series slices of raw targets
├── mlr/                        # Artifacts from MLR experiments
│   ├── <pollutant>/            # Ozone/, PM2.5/, NO2/ (expert pipelines)
│   │   ├── density_scatter_*.png
│   │   ├── residuals_*.png
│   │   ├── feature_importance_*.png
│   │   ├── mlr_model_*.pkl
│   │   └── metrics.json
├── cnn_lstm/                   # Artifacts from CNN-LSTM experiments
│   ├── training_history.png
│   ├── density_scatter/
│   ├── error_histograms/
│   └── time_series_slice.png
├── spatial_maps/
│   ├── mlr/
│   └── cnn_lstm/
├── checkpoints/                # Saved Keras models (auto-resumed)
└── logs/                       # TensorBoard logs (scalars, graphs)  
```

You can safely delete sub-folders to free disk space; future runs will recreate them as needed.

---

## 9. Citation & License

If you use this code, please cite appropriately. The project is released under the MIT License – see `LICENSE` for details. 