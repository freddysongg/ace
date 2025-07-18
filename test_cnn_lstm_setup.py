#!/usr/bin/env python3
"""
Quick validation script to test CNN+LSTM per-pollutant setup
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('.')

from src import data_loader

def test_data_loading():
    """Test data loading and preprocessing."""
    print("Testing data loading and preprocessing...")
    
    # Load data
    data_path = Path("data/input_with_geo_and_interactions_v4.npy")
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return False
    
    raw_data = np.load(data_path)
    print(f"✓ Raw data loaded: {raw_data.shape}")
    
    # Split data
    train_data, val_data, test_data = data_loader.chronological_split(
        raw_data, year_column_index=2
    )
    print(f"✓ Data split - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    
    # Define columns
    feature_indices = list(range(3, raw_data.shape[1] - 3))
    target_indices = [raw_data.shape[1] - 3, raw_data.shape[1] - 2, raw_data.shape[1] - 1]
    
    print(f"✓ Feature columns: {len(feature_indices)}")
    print(f"✓ Target columns: {target_indices}")
    
    # Test per-pollutant preprocessing
    pollutant_configs = {
        "Ozone": {"target_index": 0, "use_robust_scaler_targets": False},
        "PM2.5": {"target_index": 1, "use_robust_scaler_targets": True},
        "NO2": {"target_index": 2, "use_robust_scaler_targets": True}
    }
    
    for pollutant_name, config in pollutant_configs.items():
        print(f"\nTesting {pollutant_name} preprocessing...")
        
        try:
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
            
            print(f"  ✓ Processed data shapes:")
            print(f"    X_train: {single_pollutant_data['X_train'].shape}")
            print(f"    y_train: {single_pollutant_data['y_train'].shape}")
            print(f"    X_val: {single_pollutant_data['X_val'].shape}")
            print(f"    y_val: {single_pollutant_data['y_val'].shape}")
            
            # Test sequence creation
            X_train_seq, y_train_seq = data_loader.create_lookback_sequences(
                single_pollutant_data["X_train"],
                single_pollutant_data["y_train"],
                lookback=7,
                step=1
            )
            
            print(f"  ✓ Sequence shapes:")
            print(f"    X_train_seq: {X_train_seq.shape}")
            print(f"    y_train_seq: {y_train_seq.shape}")
            
        except Exception as e:
            print(f"  ✗ Error processing {pollutant_name}: {e}")
            return False
    
    return True

def test_model_import():
    """Test model imports."""
    print("\nTesting model imports...")
    
    try:
        from src.models import get_cnn_lstm_model
        print("✓ CNN+LSTM model import successful")
        
        # Test model creation
        input_shape = (7, 100)  # Example shape
        num_outputs = 1
        model = get_cnn_lstm_model(input_shape, num_outputs)
        print(f"✓ Model created with input shape {input_shape} and {num_outputs} outputs")
        print(f"  Model parameters: {model.count_params():,}")
        
        return True
    except Exception as e:
        print(f"✗ Model import error: {e}")
        return False

def test_training_function():
    """Test training function import."""
    print("\nTesting training function import...")
    
    try:
        from src.train import train_single_pollutant_cnn_lstm_model
        print("✓ Single-pollutant CNN+LSTM training function imported successfully")
        return True
    except Exception as e:
        print(f"✗ Training function import error: {e}")
        return False

def main():
    """Run all tests."""
    print("CNN+LSTM Per-Pollutant Setup Validation")
    print("=" * 50)
    
    tests = [
        ("Data Loading & Preprocessing", test_data_loading),
        ("Model Import", test_model_import),
        ("Training Function", test_training_function),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All tests passed! Ready to run CNN+LSTM per-pollutant training.")
        print("\nTo start training, run:")
        print("  ./run_cnn_lstm_per_pollutant.sh")
    else:
        print("❌ Some tests failed. Please fix the issues before running training.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)