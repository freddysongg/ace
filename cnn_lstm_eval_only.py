#!/usr/bin/env python3

import argparse
from pathlib import Path
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import mlflow
import tensorflow as tf

from src import (
    data_loader,
    evaluate,
)

from src.utils import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/input_with_geo_and_interactions_v4.npy",
    )
    parser.add_argument(
        "--year-column",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="test_results/cnn_lstm_per_pollutant",
    )
    
    return parser.parse_args()


def find_latest_model_dir(base_dir: Path, pollutant_name: str) -> Path:
    pollutant_dir = base_dir / pollutant_name
    if not pollutant_dir.exists():
        raise FileNotFoundError(f"No saved models found for {pollutant_name} in {pollutant_dir}")
    
    run_dirs = [d for d in pollutant_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found for {pollutant_name}")
    
    latest_run_dir = max(run_dirs, key=lambda x: x.name)
    return latest_run_dir


def main():
    args = parse_args()
    
    # Set reproducibility
    set_global_seed(42)
    
    print("CNN+LSTM Per-Pollutant Evaluation Only")
    print("=" * 50)
    
    # Data loading
    print("Loading and preprocessing data...")
    data_path = Path(args.data)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load raw data
    raw_data = np.load(data_path)
    print(f"Raw data shape: {raw_data.shape}")
    
    # Split data
    train_data, val_data, test_data = data_loader.chronological_split(
        raw_data, year_column_index=args.year_column
    )
    
    # Define feature and target columns
    feature_indices = list(range(3, raw_data.shape[1] - 3))  # Exclude first 3 and last 3
    target_indices = [raw_data.shape[1] - 3, raw_data.shape[1] - 2, raw_data.shape[1] - 1]  # Last 3 columns
    
    print(f"Feature columns: {len(feature_indices)} features")
    print(f"Target columns: {target_indices} (Ozone, PM2.5, NO2)")
    
    # Per-pollutant configuration
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
    
    # Model base directory
    model_base_dir = Path(args.model_dir)
    
    # MLflow setup
    mlflow.set_experiment("CNN_LSTM_Per_Pollutant_Evaluation")
    
    with mlflow.start_run(run_name=f"CNN_LSTM_eval_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("model_type", "cnn_lstm_per_pollutant_eval_only")
        mlflow.log_param("data_file", str(data_path))
        mlflow.log_param("model_dir", str(model_base_dir))
        
        print("\n==================== Loading Models ====================")
        
        # Storage for all results
        all_pollutant_models = {}
        all_pollutant_data = {}
        all_pollutant_metrics = {}
        
        # Load existing models for evaluation
        for pollutant_name in pollutant_configs.keys():
            print(f"\nLoading saved {pollutant_name} CNN+LSTM model...")
            
            # Find the most recent model directory for this pollutant
            latest_run_dir = find_latest_model_dir(model_base_dir, pollutant_name)
            model_path = latest_run_dir / "cnn_lstm_model.keras"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"Loading model from {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            all_pollutant_models[pollutant_name] = model
            
            # Preprocess data for this specific pollutant
            print(f"Preprocessing data for {pollutant_name}...")
            config = pollutant_configs[pollutant_name]
            single_pollutant_data = data_loader.preprocess_data(
                train_data,
                val_data,
                test_data,
                feature_columns=feature_indices,
                target_columns=target_indices,
                target_column_index=config['target_index'],
                log_transform_targets=None,
                use_robust_scaler_targets=config['use_robust_scaler_targets'],
            )
            
            # Reshape data for CNN+LSTM (single timestep)
            X_val_reshaped = single_pollutant_data["X_val"].reshape(
                single_pollutant_data["X_val"].shape[0], 1, single_pollutant_data["X_val"].shape[1]
            )
            X_test_reshaped = single_pollutant_data["X_test"].reshape(
                single_pollutant_data["X_test"].shape[0], 1, single_pollutant_data["X_test"].shape[1]
            )
            
            all_pollutant_data[pollutant_name] = {
                'processed_data': single_pollutant_data,
                'reshaped_data': {
                    'X_val': X_val_reshaped,
                    'X_test': X_test_reshaped
                },
                'results_dir': latest_run_dir
            }
            
            print(f"✓ {pollutant_name} model and data loaded successfully")
        
        print(f"\n{'='*60}")
        print("EVALUATION PHASE")
        print(f"{'='*60}")
        
        # Evaluation phase - combine predictions from all pollutants
        pollutant_names = ["Ozone", "PM2.5", "NO2"]
        
        y_pred_val_combined = []
        y_pred_test_combined = []
        y_val_raw_combined = []
        y_test_raw_combined = []
        
        for pollutant_name in pollutant_names:
            print(f"\nEvaluating {pollutant_name} model...")
            
            model = all_pollutant_models[pollutant_name]
            single_pollutant_data = all_pollutant_data[pollutant_name]['processed_data']
            reshaped_data = all_pollutant_data[pollutant_name]['reshaped_data']
            results_dir = all_pollutant_data[pollutant_name]['results_dir']
            
            # Get predictions
            print(f"Making predictions for {pollutant_name}...")
            y_pred_val_single = model.predict(reshaped_data['X_val'], verbose=0)
            y_pred_test_single = model.predict(reshaped_data['X_test'], verbose=0)
            
            # Ensure predictions are 1D for single-pollutant models
            if y_pred_val_single.ndim > 1:
                y_pred_val_single = y_pred_val_single.ravel()
            if y_pred_test_single.ndim > 1:
                y_pred_test_single = y_pred_test_single.ravel()
            
            # Transform back to original scale
            target_scaler = single_pollutant_data["target_scaler"]
            y_pred_val_orig = target_scaler.inverse_transform(y_pred_val_single.reshape(-1, 1)).ravel()
            y_pred_test_orig = target_scaler.inverse_transform(y_pred_test_single.reshape(-1, 1)).ravel()
            
            # Get raw targets (no sequence trimming needed)
            y_val_raw_single = single_pollutant_data["y_val_raw"].ravel()
            y_test_raw_single = single_pollutant_data["y_test_raw"].ravel()
            
            # Store for combined evaluation
            y_pred_val_combined.append(y_pred_val_orig)
            y_pred_test_combined.append(y_pred_test_orig)
            y_val_raw_combined.append(y_val_raw_single)
            y_test_raw_combined.append(y_test_raw_single)
            
            # Generate individual pollutant visualizations
            print(f"Generating visualizations for {pollutant_name}...")
            
            # Create evaluation subdirectory
            eval_dir = results_dir / "evaluation"
            eval_dir.mkdir(exist_ok=True)
            
            # Density scatter plot
            n_samples_total = len(y_test_raw_single)
            sample_size = min(5000, n_samples_total)
            if sample_size < n_samples_total:
                sample_idx = np.random.choice(n_samples_total, sample_size, replace=False)
                y_test_sample = y_test_raw_single[sample_idx]
                y_pred_sample = y_pred_test_orig[sample_idx]
            else:
                y_test_sample = y_test_raw_single
                y_pred_sample = y_pred_test_orig
            
            evaluate.density_scatter_plot(
                y_test_sample,
                y_pred_sample,
                pollutant_name=pollutant_name,
                save_path=str(eval_dir / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_density_scatter.png"),
                show=False
            )
            
            # Residuals plot
            evaluate.residuals_plot(
                y_test_sample,
                y_pred_sample,
                pollutant_name=pollutant_name,
                save_path=str(eval_dir / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_residuals.png"),
                show=False
            )
            
            print(f"✓ Visualizations saved for {pollutant_name}")
        
        # Combine all predictions for overall evaluation
        print("\nCombining predictions for overall evaluation...")
        
        # Find minimum length across all pollutants
        min_val_len = min(len(arr) for arr in y_pred_val_combined)
        min_test_len = min(len(arr) for arr in y_pred_test_combined)
        
        # Trim all arrays to minimum length
        y_pred_val_combined = [arr[:min_val_len] for arr in y_pred_val_combined]
        y_val_raw_combined = [arr[:min_val_len] for arr in y_val_raw_combined]
        y_pred_test_combined = [arr[:min_test_len] for arr in y_pred_test_combined]
        y_test_raw_combined = [arr[:min_test_len] for arr in y_test_raw_combined]
        
        # Stack to create multi-pollutant arrays
        y_pred_val_orig = np.column_stack(y_pred_val_combined)
        y_pred_test_orig = np.column_stack(y_pred_test_combined)
        y_val_raw = np.column_stack(y_val_raw_combined)
        y_test_raw = np.column_stack(y_test_raw_combined)
        
        print(f"Combined prediction shapes - Val: {y_pred_val_orig.shape}, Test: {y_pred_test_orig.shape}")
        
        # Calculate comprehensive metrics
        print("\nCalculating comprehensive metrics...")
        val_metrics = evaluate.calculate_summary_metrics(
            y_val_raw, y_pred_val_orig, pollutant_names
        )
        test_metrics = evaluate.calculate_summary_metrics(
            y_test_raw, y_pred_test_orig, pollutant_names
        )
        
        # Store metrics for each pollutant
        for pollutant in pollutant_names:
            all_pollutant_metrics[pollutant] = {
                "validation": val_metrics[pollutant],
                "test": test_metrics[pollutant]
            }
        
        # Display results with enhanced formatting
        print("\n" + "="*60)
        print("FINAL EVALUATION METRICS")
        print("="*60)
        
        for pollutant in pollutant_names:
            print(f"\n--- {pollutant} ---")
            # Validation Metrics
            print(f"  Validation RMSE: {val_metrics[pollutant]['RMSE']:.4f}")
            print(f"  Validation R²:   {val_metrics[pollutant]['R2']:.4f}")
            print(f"  Validation MAE:  {val_metrics[pollutant]['MAE']:.4f}")
            print(f"  Validation Bias: {val_metrics[pollutant]['Bias']:.4f}")
            
            # Test Metrics
            print(f"  Test R²:         {test_metrics[pollutant]['R2']:.4f}")
            print(f"  Test RMSE:       {test_metrics[pollutant]['RMSE']:.2f}")
            print(f"  Test MAE:        {test_metrics[pollutant]['MAE']:.2f}")
            print(f"  Test Bias:       {test_metrics[pollutant]['Bias']:.2f}")
            
            # Normalized Metrics
            print("  ----------- Normalized -----------")
            nrmse_pct = test_metrics[pollutant].get('NRMSE', float('nan')) * 100
            cv_rmse_pct = test_metrics[pollutant].get('CV_RMSE', float('nan')) * 100
            norm_mae_pct = test_metrics[pollutant].get('Norm_MAE', float('nan')) * 100
            norm_bias_pct = test_metrics[pollutant].get('Norm_Bias', float('nan')) * 100
            
            print(f"  NRMSE (% of Range):   {nrmse_pct:.2f}%")
            print(f"  CV(RMSE) (% of Mean): {cv_rmse_pct:.2f}%")
            print(f"  Norm MAE (% of Mean): {norm_mae_pct:.2f}%")
            print(f"  Norm Bias (% of Mean):{norm_bias_pct:+.2f}%")
        
        # Save evaluation results
        print("\n===== Saving Evaluation Results =====")
        
        # Create evaluation results directory
        eval_results_dir = Path("test_results/cnn_lstm_evaluation_results")
        eval_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save comprehensive evaluation report
        evaluation_report = {
            "timestamp": datetime.now().isoformat(),
            "model_type": "cnn_lstm_per_pollutant_evaluation",
            "data_file": str(data_path),
            "model_directories": {
                pollutant: str(all_pollutant_data[pollutant]['results_dir'])
                for pollutant in pollutant_names
            },
            "pollutant_metrics": {
                pollutant: {
                    "validation": convert_numpy_types(val_metrics[pollutant]),
                    "test": convert_numpy_types(test_metrics[pollutant])
                }
                for pollutant in pollutant_names
            },
            "aggregate_metrics": {
                "avg_val_rmse": float(np.mean([val_metrics[p]['RMSE'] for p in pollutant_names])),
                "avg_val_r2": float(np.mean([val_metrics[p]['R2'] for p in pollutant_names])),
                "avg_test_rmse": float(np.mean([test_metrics[p]['RMSE'] for p in pollutant_names])),
                "avg_test_r2": float(np.mean([test_metrics[p]['R2'] for p in pollutant_names]))
            },
            "normalized_metrics_explanation": {
                "NRMSE": "Normalized RMSE (RMSE / range of true values)",
                "CV_RMSE": "Coefficient of Variation of RMSE (RMSE / mean of true values)",
                "Norm_MAE": "Normalized MAE (MAE / mean of true values)",
                "Norm_Bias": "Normalized Bias (Bias / mean of true values)"
            }
        }
        
        with open(eval_results_dir / "evaluation_report.json", 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        print(f"Evaluation report saved to {eval_results_dir / 'evaluation_report.json'}")
        
        # MLflow logging
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
            
            # Normalized metrics
            mlflow.log_metric(f"test_nrmse_{pollutant_key}", test_metrics[pollutant].get('NRMSE', float('nan')))
            mlflow.log_metric(f"test_cv_rmse_{pollutant_key}", test_metrics[pollutant].get('CV_RMSE', float('nan')))
            mlflow.log_metric(f"test_norm_mae_{pollutant_key}", test_metrics[pollutant].get('Norm_MAE', float('nan')))
            mlflow.log_metric(f"test_norm_bias_{pollutant_key}", test_metrics[pollutant].get('Norm_Bias', float('nan')))
            
            print(f"✓ Logged metrics for {pollutant}")
        
        # Log aggregate metrics
        avg_val_rmse = np.mean([val_metrics[p]['RMSE'] for p in pollutant_names])
        avg_val_r2 = np.mean([val_metrics[p]['R2'] for p in pollutant_names])
        avg_test_rmse = np.mean([test_metrics[p]['RMSE'] for p in pollutant_names])
        avg_test_r2 = np.mean([test_metrics[p]['R2'] for p in pollutant_names])
        
        mlflow.log_metric("avg_val_rmse", avg_val_rmse)
        mlflow.log_metric("avg_val_r2", avg_val_r2)
        mlflow.log_metric("avg_test_rmse", avg_test_rmse)
        mlflow.log_metric("avg_test_r2", avg_test_r2)
        
        print("✓ Aggregate metrics logged to MLflow")
        
        # Generate combined visualizations
        print("\nGenerating combined visualizations...")
        combined_results_dir = eval_results_dir / "combined_visualizations"
        combined_results_dir.mkdir(exist_ok=True)
        
        # Multi-pollutant density scatter plots
        n_samples_total = y_test_raw.shape[0]
        sample_size = min(5000, n_samples_total)
        if sample_size < n_samples_total:
            sample_idx = np.random.choice(n_samples_total, sample_size, replace=False)
            y_test_sample = y_test_raw[sample_idx]
            y_pred_sample = y_pred_test_orig[sample_idx]
        else:
            y_test_sample = y_test_raw
            y_pred_sample = y_pred_test_orig
        
        evaluate.density_scatter_plots_multi(
            y_test_sample,
            y_pred_sample,
            pollutant_names=pollutant_names,
            save_dir=str(combined_results_dir / "density_scatter"),
            show=False
        )
        
        # Multi-pollutant error histograms
        evaluate.prediction_error_histograms_multi(
            y_test_raw,
            y_pred_test_orig,
            pollutant_names=pollutant_names,
            save_dir=str(combined_results_dir / "error_histograms"),
            show=False
        )
        
        print(f"✓ Combined visualizations saved to {combined_results_dir}")
        
        print("\n" + "="*60)
        print("CNN+LSTM PER-POLLUTANT EVALUATION COMPLETED")
        print("="*60)
        print(f"Evaluation results saved in: {eval_results_dir}")
        print(f"Individual model results: {model_base_dir}")
        print("="*60)


if __name__ == "__main__":
    main()