"""
src/models/ann_model.py
Keras ANN with BatchNormalization, Dropout, EarlyStopping, and ReduceLROnPlateau.
ELU activations are used to avoid dying-neuron issues on small datasets.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.metrics import compute_metrics
from config import (
    ANN_HIDDEN_LAYERS, ANN_EPOCHS, ANN_BATCH_SIZE,
    ANN_LEARNING_RATE, ANN_DROPOUT, ANN_L2, ANN_PATIENCE, RANDOM_SEED
)


def build_ann(input_dim,
              hidden_layers=ANN_HIDDEN_LAYERS,
              dropout_rate=ANN_DROPOUT,
              l2_reg=ANN_L2,
              learning_rate=ANN_LEARNING_RATE):
    """
    Build a feedforward ANN.

    Architecture:
        Input
        → Dense(64, ELU) → BatchNorm → Dropout
        → Dense(32, ELU) → BatchNorm → Dropout
        → Dense(16, ELU)
        → Dense(1, linear)
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers
    except ImportError:
        raise ImportError("tensorflow not installed. Run: pip install tensorflow")

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(
            units, activation='elu',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(l2_reg),
        ))
        if i < len(hidden_layers) - 1:
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )
    return model


def train_ann(X_train_s, X_test_s, y_train_s, y_test_s,
              scaler_y,
              hidden_layers=ANN_HIDDEN_LAYERS,
              epochs=ANN_EPOCHS,
              batch_size=ANN_BATCH_SIZE,
              learning_rate=ANN_LEARNING_RATE,
              dropout_rate=ANN_DROPOUT,
              l2_reg=ANN_L2,
              patience=ANN_PATIENCE,
              random_seed=RANDOM_SEED):
    """
    Train the ANN with EarlyStopping and ReduceLROnPlateau.

    Returns
    -------
    model        : trained Keras model
    history      : training History object
    metrics      : dict {R2, RMSE, MAE, epochs_trained, best_val_loss}
    y_pred_orig  : predictions in original scale
    y_test_orig  : true values in original scale
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import callbacks
    except ImportError:
        raise ImportError("tensorflow not installed. Run: pip install tensorflow")

    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    model = build_ann(
        input_dim=X_train_s.shape[1],
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        learning_rate=learning_rate,
    )

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=patience,
            restore_best_weights=True, verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=patience // 2, min_lr=1e-6, verbose=0,
        ),
    ]

    history = model.fit(
        X_train_s, y_train_s,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=0,
    )

    y_pred_s   = model.predict(X_test_s, verbose=0).ravel()
    y_pred_orig = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_test_orig = scaler_y.inverse_transform(y_test_s.reshape(-1, 1)).ravel()

    metrics = {
        **compute_metrics(y_test_orig, y_pred_orig),
        'epochs_trained':  len(history.history['loss']),
        'best_val_loss':   float(min(history.history['val_loss'])),
    }

    return model, history, metrics, y_pred_orig, y_test_orig


def save_ann(model, path):
    """Save Keras model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"[ANN] Model saved -> {path}")


def load_ann(path):
    """Load a saved Keras model."""
    try:
        from tensorflow import keras
        return keras.models.load_model(path)
    except Exception as e:
        print(f"[ANN] Could not load model from {path}: {e}")
        return None
