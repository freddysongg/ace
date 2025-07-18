#!/usr/bin/env python3

import argparse
from pathlib import Path
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gc
from datetime import datetime

import mlflow
import tensorflow as tf

from src import (
    data_loader,
    train,
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
        "--no-resume",
        action="store_true",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set reproducibility
    set_global_seed(42)
    
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
            "log_transform": False,
            "epochs": args.epochs,
            "batch_size": args.batch_size
        },
        "PM2.5": {
            "target_index": 1,
            "use_robust_scaler_targets": True,   # RobustScaler for outliers
            "log_transform": False,
            "epochs": args.epochs,
            "batch_size": args.batch_size
        },
        "NO2": {
            "target_index": 2,
            "use_robust_scaler_targets": True,   # RobustScaler for outliers
            "log_transform": False,
            "epochs": args.epochs,
            "batch_size": args.batch_size
        }
    }
    
    # Create base results directory
    base_results_dir = Path("test_results")
    base_results_dir.mkdir(exist_ok=True)
    
    # MLflow setup
    mlflow.set_experiment("CNN_LSTM_Per_Pollutant_Training")
    
    with mlflow.start_run(run_name=f"CNN_LSTM_per_pollutant_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log global parameters
        mlflow.log_param("model_type", "cnn_lstm_per_pollutant_no_lookback")
        mlflow.log_param("data_file", str(data_path))
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("eval_only", args.eval_only)
        
        # Storage for all results
        all_pollutant_models = {}
        all_pollutant_histories = {}
        all_pollutant_data = {}
        all_pollutant_metrics = {}
        
        if not args.eval_only:
            print("\n==================== Training Mode ====================")
            print("Starting per-pollutant CNN+LSTM training loop...")
            
            for pollutant_name, config in pollutant_configs.items():
                print(f"\n{'='*60}")
                print(f"Training {pollutant_name} CNN+LSTM Model")
                print(f"{'='*60}")
                
                # Create results directory for this pollutant
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pollutant_results_dir = base_results_dir / "cnn_lstm_per_pollutant" / pollutant_name / f"run_{timestamp}"
                pollutant_results_dir.mkdir(parents=True, exist_ok=True)
                print(f"Results will be saved to: {pollutant_results_dir}")
                
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
                
                print(f"Training data shape for {pollutant_name}: {single_pollutant_data['X_train'].shape}")
                print(f"Target data shape for {pollutant_name}: {single_pollutant_data['y_train'].shape}")
                print(f"Using {'RobustScaler' if config['use_robust_scaler_targets'] else 'StandardScaler'} for {pollutant_name}")
                
                # Reshape 2D data to 3D for CNN+LSTM (treating each sample as a single timestep)
                print(f"Reshaping data for CNN+LSTM architecture...")
                X_train_reshaped = single_pollutant_data["X_train"].reshape(
                    single_pollutant_data["X_train"].shape[0], 1, single_pollutant_data["X_train"].shape[1]
                )
                X_val_reshaped = single_pollutant_data["X_val"].reshape(
                    single_pollutant_data["X_val"].shape[0], 1, single_pollutant_data["X_val"].shape[1]
                )
                X_test_reshaped = single_pollutant_data["X_test"].reshape(
                    single_pollutant_data["X_test"].shape[0], 1, single_pollutant_data["X_test"].shape[1]
                )
                
                print(f"Reshaped data - Train: {X_train_reshaped.shape}, Val: {X_val_reshaped.shape}, Test: {X_test_reshaped.shape}")
                
                # Train CNN+LSTM model
                print(f"Training CNN+LSTM model for {pollutant_name}...")
                model, history = train.train_single_pollutant_cnn_lstm_model(
                    X_train_reshaped,
                    single_pollutant_data["y_train"].ravel(),  # Convert to 1D for single-pollutant training
                    X_val_reshaped,
                    single_pollutant_data["y_val"].ravel(),    # Convert to 1D for single-pollutant training
                    pollutant_name=pollutant_name,
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    resume=not args.no_resume,
                    use_generator=False  # Use arrays directly for per-pollutant training
                )
                
                # Store model, history, and data
                all_pollutant_models[pollutant_name] = model
                all_pollutant_histories[pollutant_name] = history
                all_pollutant_data[pollutant_name] = {
                    'processed_data': single_pollutant_data,
                    'reshaped_data': {
                        'X_train': X_train_reshaped,
                        'X_val': X_val_reshaped,
                        'X_test': X_test_reshaped
                    },
                    'results_dir': pollutant_results_dir
                }
                
                # Save model
                model_path = pollutant_results_dir / "cnn_lstm_model.keras"
                model.save(model_path)
                print(f"Model saved to {model_path}")
                
                # Save training history
                if history and hasattr(history, 'history') and history.history:
                    # Save training history plot
                    evaluate.training_history_plot(
                        history,
                        save_path=str(pollutant_results_dir / "training_history.png"),
                        show=False,
                        title=f"CNN+LSTM Training History - {pollutant_name}"
                    )
                    
                    # Save history as JSON
                    def _py(val):
                        if isinstance(val, (np.floating, np.integer)):
                            return val.item()
                        return val
                    
                    hist_serializable = {
                        k: [_py(v) for v in vals] for k, vals in history.history.items()
                    }
                    with open(pollutant_results_dir / "training_history.json", "w") as f:
                        json.dump(hist_serializable, f, indent=2)
                
                # Save configuration
                config_data = {
                    "pollutant_name": pollutant_name,
                    "timestamp": datetime.now().isoformat(),
                    "configuration": config,
                    "data_shapes": {
                        "original_train": str(single_pollutant_data['X_train'].shape),
                        "reshaped_train": str(X_train_reshaped.shape),
                        "target_train": str(single_pollutant_data['y_train'].shape)
                    }
                }
                
                with open(pollutant_results_dir / "config_validation.json", 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                print(f"Completed training for {pollutant_name}")
                
                # Memory cleanup
                gc.collect()
                tf.keras.backend.clear_session()
        
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
            
            config = pollutant_configs[pollutant_name]
            model = all_pollutant_models[pollutant_name]
            
            # Get data
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
                save_path=str(results_dir / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_density_scatter.png"),
                show=False
            )
            
            # Residuals plot
            evaluate.residuals_plot(
                y_test_sample,
                y_pred_sample,
                pollutant_name=pollutant_name,
                save_path=str(results_dir / f"{pollutant_name.lower().replace('.', '').replace(' ', '_')}_residuals.png"),
                show=False
            )
            
            print(f"Visualizations saved for {pollutant_name}")
        
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
        
        # Save individual pollutant metrics reports
        for pollutant in pollutant_names:
            results_dir = all_pollutant_data[pollutant]['results_dir']
            
            # Save metrics report
            metrics_file = results_dir / "metrics_report.json"
            
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
            
            report = {
                "pollutant_name": pollutant,
                "timestamp": datetime.now().isoformat(),
                "validation_metrics": convert_numpy_types(all_pollutant_metrics[pollutant]["validation"]),
                "test_metrics": convert_numpy_types(all_pollutant_metrics[pollutant]["test"]),
                "normalized_metrics_explanation": {
                    "NRMSE": "Normalized RMSE (RMSE / range of true values)",
                    "CV_RMSE": "Coefficient of Variation of RMSE (RMSE / mean of true values)",
                    "Norm_MAE": "Normalized MAE (MAE / mean of true values)",
                    "Norm_Bias": "Normalized Bias (Bias / mean of true values)"
                }
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save training summary
            summary_file = results_dir / "training_summary.json"
            summary_data = {
                "pollutant_name": pollutant,
                "model_type": "cnn_lstm_no_lookback",
                "training_completed": True,
                "final_metrics": all_pollutant_metrics[pollutant],
                "configuration": pollutant_configs[pollutant]
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
        
        # MLflow logging
        print("\n===== MLflow Logging =====")
        
        # Log individual pollutant metrics
        for pollutant in pollutant_names:
            pollutant_key = pollutant.lower().replace(".", "").replace(" ", "_")
            
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
            
            print(f"Logged metrics for {pollutant}")
        
        # Log aggregate metrics
        avg_test_rmse = np.mean([test_metrics[p]['RMSE'] for p in pollutant_names])
        avg_test_r2 = np.mean([test_metrics[p]['R2'] for p in pollutant_names])
        avg_test_mae = np.mean([test_metrics[p]['MAE'] for p in pollutant_names])
        avg_test_bias = np.mean([test_metrics[p]['Bias'] for p in pollutant_names])
        
        mlflow.log_metric("avg_test_rmse", avg_test_rmse)
        mlflow.log_metric("avg_test_r2", avg_test_r2)
        mlflow.log_metric("avg_test_mae", avg_test_mae)
        mlflow.log_metric("avg_test_bias", avg_test_bias)
        
        print("Aggregate metrics logged to MLflow")
        
        # Save final combined results
        combined_results_dir = base_results_dir / "cnn_lstm_combined_results"
        combined_results_dir.mkdir(exist_ok=True)
        
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "model_type": "cnn_lstm_per_pollutant_no_lookback",
            "training_mode": "full_training",
            "pollutants": pollutant_names,
            "configurations": pollutant_configs,
            "final_metrics": {
                "validation": {p: val_metrics[p] for p in pollutant_names},
                "test": {p: test_metrics[p] for p in pollutant_names}
            },
            "aggregate_metrics": {
                "avg_test_rmse": avg_test_rmse,
                "avg_test_r2": avg_test_r2,
                "avg_test_mae": avg_test_mae,
                "avg_test_bias": avg_test_bias
            }
        }
        
        with open(combined_results_dir / "final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\nFinal results saved to {combined_results_dir / 'final_results.json'}")
        
        print("\n" + "="*60)
        print("CNN+LSTM PER-POLLUTANT TRAINING COMPLETED")
        print("="*60)
        print(f"Results saved in: {base_results_dir / 'cnn_lstm_per_pollutant'}")
        print(f"Combined results: {combined_results_dir}")
        print("="*60)


if __name__ == "__main__":
    main()