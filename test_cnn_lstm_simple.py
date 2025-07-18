#!/usr/bin/env python3
"""
Simple test of CNN+LSTM training without lookback sequences
"""

import numpy as np
import tensorflow as tf
from src.models import get_cnn_lstm_model_no_pooling

def test_cnn_lstm_no_pooling():
    """Test the CNN+LSTM model without pooling."""
    print("Testing CNN+LSTM model without pooling...")
    
    # Create test data
    batch_size = 32
    timesteps = 1
    features = 46
    
    X_train = np.random.randn(100, timesteps, features)
    y_train = np.random.randn(100)
    X_val = np.random.randn(20, timesteps, features)
    y_val = np.random.randn(20)
    
    print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Create model
    model = get_cnn_lstm_model_no_pooling((timesteps, features), 1)
    print(f"Model created: {model.name}")
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    print("Model compiled successfully")
    
    # Test prediction before training
    pred_before = model.predict(X_val[:5], verbose=0)
    print(f"Prediction before training: {pred_before.shape}")
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=2,
        batch_size=batch_size,
        verbose=1
    )
    
    # Test prediction after training
    pred_after = model.predict(X_val[:5], verbose=0)
    print(f"Prediction after training: {pred_after.shape}")
    
    print("SUCCESS: CNN+LSTM training completed without errors!")
    return model, history

if __name__ == "__main__":
    try:
        model, history = test_cnn_lstm_no_pooling()
        print("\n✅ Test passed! CNN+LSTM model works with single timesteps.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()