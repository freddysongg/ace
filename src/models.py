"""
Models module for air pollutant prediction.

This module contains model definitions for:
- Multiple Linear Regression (MLR)
- CNN+LSTM (to be implemented)
"""

from sklearn.linear_model import LinearRegression
from tensorflow.keras import layers, models, regularizers
from typing import Optional


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


def get_mlp_model(
    input_dim: int,
    num_outputs: int = 3,
    hidden_units: tuple = (64, 32),
    dropout: float = 0.3,
    l2_reg: Optional[float] = None,
):
    inputs = layers.Input(shape=(input_dim,), name="input_layer")
    x = inputs
    for i, units in enumerate(hidden_units):
        kernel_reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=kernel_reg,
            name=f"dense_{i+1}",
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i+1}")(x)

    kernel_reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    outputs = layers.Dense(
        num_outputs,
        activation="linear",
        kernel_regularizer=kernel_reg,
        name="output_layer",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="mlp_air_pollutant")
    return model


def get_single_pollutant_mlp_model(
    input_dim: int,
    hidden_units: tuple = (64, 32),
    dropout: float = 0.3,
    l2_reg: Optional[float] = None,
):
    """
    Create and return an MLP model for single pollutant prediction.
    
    This function creates a model identical to get_mlp_model() but with num_outputs=1
    for training separate models per pollutant.
    
    Args:
        input_dim (int): Number of input features
        hidden_units (tuple): Tuple of hidden layer sizes (default: (64, 32))
        dropout (float): Dropout rate for regularization (default: 0.3)
        l2_reg (Optional[float]): L2 regularization strength (default: None)
    
    Returns:
        tensorflow.keras.Model: An untrained MLP model with single output
    """
    inputs = layers.Input(shape=(input_dim,), name="input_layer")
    x = inputs
    for i, units in enumerate(hidden_units):
        kernel_reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=kernel_reg,
            name=f"dense_{i+1}",
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i+1}")(x)

    kernel_reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    outputs = layers.Dense(
        1,  # Single output for single pollutant
        activation="linear",
        kernel_regularizer=kernel_reg,
        name="output_layer",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="single_pollutant_mlp")
    return model


def get_simple_lstm_model(
    input_shape: tuple,
    num_outputs: int = 3,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.3,
):

    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = layers.LSTM(lstm_units, return_sequences=False, name="lstm")(inputs)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout_2")(x)
    outputs = layers.Dense(num_outputs, activation="linear", name="output_layer")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="simple_lstm_air_pollutant")
    return model

def get_cnn_lstm_model_no_pooling(input_shape, num_outputs=3):
    """
    Create and return a CNN+LSTM model for single timestep data (no pooling layers).
    
    This version is designed to work with input_shape like (1, features) where
    the temporal dimension is 1, avoiding issues with MaxPooling1D.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features)
        num_outputs (int): Number of output targets (default: 3 for Ozone, PM2.5, NO2)

    Returns:
        tensorflow.keras.Model: An untrained CNN+LSTM model without pooling
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # Convolutional layers for feature extraction (no pooling to avoid dimension issues)
    x = layers.Conv1D(filters=32, kernel_size=1, activation="relu", padding="same")(inputs)
    x = layers.Conv1D(filters=64, kernel_size=1, activation="relu", padding="same")(x)
    
    # LSTM layer to capture dependencies (even with single timestep)
    x = layers.LSTM(64, return_sequences=False)(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.5)(x)

    # Fully connected layers
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_outputs, activation="linear", name="output_layer")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_no_pooling")
    return model