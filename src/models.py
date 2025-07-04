"""
Models module for air pollutant prediction.

This module contains model definitions for:
- Multiple Linear Regression (MLR)
- CNN+LSTM (to be implemented)
"""

from sklearn.linear_model import LinearRegression
from tensorflow.keras import layers, models


def get_mlr_model():
    """
    Create and return a Multiple Linear Regression model.

    Returns:
        sklearn.linear_model.LinearRegression: An untrained LinearRegression model
    """
    return LinearRegression()


def get_cnn_lstm_model(input_shape, num_outputs=3):
    """
    Create and return a CNN+LSTM model for multi-output regression.

    Args:
        input_shape (tuple): Shape of the input data
        num_outputs (int): Number of output targets (default: 3 for Ozone, PM2.5, NO2)

    Returns:
        tensorflow.keras.Model: An untrained CNN+LSTM model
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # Convolutional layers for feature extraction
    x = layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(
        inputs
    )
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # LSTM layer to capture temporal dependencies
    x = layers.LSTM(64, return_sequences=False)(x)
    # Dropout for regularization
    x = layers.Dropout(0.5)(x)

    # Fully connected layers
    x = layers.Dense(32, activation="relu")(x)
    # Additional Dropout layer
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_outputs, activation="linear", name="output_layer")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_air_pollutant")
    return model
