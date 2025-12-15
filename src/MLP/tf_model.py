# tf_model.py
import keras
import tensorflow as tf
from keras import layers, models, regularizers, callbacks
import numpy as np
from typing import Sequence, Tuple

def build_mlp(input_dim: int,
                hidden_layers: Sequence[int] = (64, 32),
                activation: str = 'relu',
                output_units: int = 7,
                output_activation: str = 'softmax',
                l2_reg: float = 1e-4,
                dropout_rate: float = 0.3) -> models.Model:
    """
    Construye un MLP Keras con capas ocultas en 'hidden_layers'.
    - input_dim: dimensión del input (n columnas tras preprocesar)
    - hidden_layers: tuplas con número de neuronas por capa
    - activation: activación en capas ocultas (recomendado: 'relu')
    - output_units: número de clases
    - output_activation: 'softmax' para clasificación multiclase
    - l2_reg: regularización L2
    - dropout_rate: dropout entre capas
    """
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units,
                            activation=activation,
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(l2_reg),
                            name=f'dense_{i+1}')(x)
        if dropout_rate and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
    outputs = layers.Dense(output_units, activation=output_activation, name='output')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def compile_model(model: models.Model, learning_rate: float = 1e-3):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, # type: ignore
                    loss='sparse_categorical_crossentropy',  # si y son ints
                    metrics=['accuracy'])
    return model

def train_model(
        model: models.Model, X_train: np.ndarray, y_train: np.ndarray,
        validation: float | tuple,
        batch_size: int = 32,
        epochs: int = 100,
        model_path: str = 'src/test/mlp_model.keras',
        patience: int = 10
    ):

    history = None

    if type(validation) is float:
        es = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        mc = callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True) 
        history = model.fit(X_train, y_train,
                            validation_split=validation ,
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=[es, mc])
    
    if type(validation) is tuple:
        X_val, y_val = validation
        es = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        mc = callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True) 
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=[es, mc])

    return history
