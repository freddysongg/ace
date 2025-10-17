"""
Models module for air pollutant prediction.

Contains neural network architectures for temporal air quality forecasting:
- CNN+LSTM models for sequence prediction
- MLP models for tabular data
- Single-pollutant specialized variants
"""

from sklearn.linear_model import LinearRegression
from tensorflow.keras import layers, models, regularizers
from typing import Optional


def get_mlr_model():
    """Create Multiple Linear Regression model for baseline comparison."""
    return LinearRegression()


def get_cnn_lstm_model(input_shape, num_outputs=3):
    """
    CNN+LSTM model for multi-output air quality prediction.
    
    Architecture combines convolutional feature extraction with LSTM temporal modeling.
    Includes regularization to prevent overfitting on temporal patterns.
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # Feature extraction with 1D convolution
    x = layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same",
                     kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same",
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Temporal sequence processing
    x = layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2,
                   kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Dense prediction layers
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_outputs, activation="linear", 
                          kernel_regularizer=regularizers.l2(0.0005), 
                          name="output_layer")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_air_pollutant_regularized")
    return model


def get_mlp_model(
    input_dim: int,
    num_outputs: int = 3,
    hidden_units: tuple = (64, 32),
    dropout: float = 0.3,
    l2_reg: Optional[float] = None,
):
    """Multi-layer perceptron for tabular air quality prediction."""
    inputs = layers.Input(shape=(input_dim,), name="input_layer")
    x = inputs
    
    # Build hidden layers with optional regularization
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

    # Output layer
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
    """Single-output MLP model for individual pollutant prediction."""
    inputs = layers.Input(shape=(input_dim,), name="input_layer")
    x = inputs
    
    # Build hidden layers
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

    # Single output layer for one pollutant
    kernel_reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    outputs = layers.Dense(
        1,
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
    """Basic LSTM model for sequence-to-value prediction."""
    inputs = layers.Input(shape=input_shape, name="input_layer")
    
    # Single LSTM layer for temporal processing
    x = layers.LSTM(lstm_units, return_sequences=False, name="lstm")(inputs)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout_1")(x)
        
    # Dense layers for final prediction
    x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout_2")(x)
    outputs = layers.Dense(num_outputs, activation="linear", name="output_layer")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="simple_lstm_air_pollutant")
    return model

def get_cnn_lstm_model_no_pooling_regularized(input_shape, num_outputs=3):
    """
    Heavily regularized CNN+LSTM model to prevent overfitting.
    
    Uses simplified architecture with strong regularization for temporal robustness.
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")
    
    # Simplified feature extraction with strong regularization
    x = layers.Conv1D(filters=32, kernel_size=1, activation="relu", padding="same", 
                     kernel_regularizer=regularizers.l2(0.01), name="conv1d_1")(inputs)
    x = layers.BatchNormalization(name="bn_conv_1")(x)
    x = layers.Dropout(0.4, name="dropout_conv_1")(x)
    
    conv_out = layers.Conv1D(filters=64, kernel_size=1, activation="relu", padding="same", 
                            kernel_regularizer=regularizers.l2(0.01), name="conv1d_2")(x)
    conv_out = layers.BatchNormalization(name="bn_conv_2")(conv_out)
    conv_out = layers.Dropout(0.3, name="dropout_conv_2")(conv_out)
    
    # Simplified LSTM with heavy regularization
    lstm_out = layers.LSTM(32, return_sequences=False, dropout=0.4, recurrent_dropout=0.3,
                          kernel_regularizer=regularizers.l2(0.01),
                          recurrent_regularizer=regularizers.l2(0.01),
                          name="lstm")(conv_out)
    
    # Dense layers with strong regularization
    x = layers.Dense(64, activation="relu", 
                    kernel_regularizer=regularizers.l2(0.01), name="dense_1")(lstm_out)
    x = layers.BatchNormalization(name="bn_dense_1")(x)
    x = layers.Dropout(0.5, name="dropout_dense_1")(x)
    
    x = layers.Dense(32, activation="relu",
                    kernel_regularizer=regularizers.l2(0.01), name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_dense_2")(x)
    x = layers.Dropout(0.4, name="dropout_dense_2")(x)
    
    outputs = layers.Dense(
        num_outputs, 
        activation="linear", 
        kernel_regularizer=regularizers.l2(0.01),
        name="output_layer"
    )(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="regularized_cnn_lstm")
    return model