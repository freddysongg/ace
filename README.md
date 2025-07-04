# Air Pollutant Prediction Models

Multi-model framework for predicting ground-level concentrations of key air pollutants (Ozone, PM2.5, NO2) across the contiguous United States.

---

## Table of Contents
1. Project Overview
2. Directory Structure
3. Installation
4. Data
5. Running the Pipeline
6. Experiment Tracking with MLflow
7. Testing
8. Results
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
└── main.py              # Pipeline entry-point
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

## 5. Running the Pipeline
Run the **complete workflow** (data → training → evaluation) with:

```bash
python main.py
```

This will:
1. Load and split the dataset (training / validation / testing).
2. Train a separate MLR model for each pollutant.
3. Train the multi-output CNN+LSTM model.
4. Generate evaluation plots and save them under `results/`.
5. Log metrics and artifacts to the current MLflow experiment.

> **CLI Options (Upcoming)** – Argument parsing to run individual models will be added in task 6.4.

---

## 6. Experiment Tracking with MLflow

Inside the project directory run:

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser to explore runs, parameters, metrics, and saved artifacts.

---

## 7. Testing

Execute unit tests with **pytest**:

```bash
pytest -q
```

---

## 8. Results
All generated figures and serialized models are stored under `results/`.

---

## 9. Citation & License

If you use this code, please cite appropriately. The project is released under the MIT License – see `LICENSE` for details. 