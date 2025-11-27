"""
Template alternativ: rețea CNN antrenată pe CIFAR-10
Compatibil cu Federated Learning și TensorFlow/Keras
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any
import json

# ============================================================================
# 1. ÎNCĂRCARE DATE CIFAR-10
# ============================================================================

def load_train_test_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Încărcare și împărțire set de date CIFAR-10
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Conversie la tf.data.Dataset
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return ds_train, ds_test


def preprocess(image, label):
    """
    Normalizează imaginile și transformă etichetele în one-hot vectors
    """
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (32, 32))
    label = tf.one_hot(tf.squeeze(label), 10)
    return image, label


def preprocess_loaded_data(train_ds, test_ds) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds_processed = train_ds.map(preprocess).shuffle(20000).batch(64).prefetch(tf.data.AUTOTUNE)
    test_ds_processed = test_ds.map(preprocess).batch(64).prefetch(tf.data.AUTOTUNE)
    return train_ds_processed, test_ds_processed

# ============================================================================
# 2. ANTRENARE REȚEA NEURONALĂ
# ============================================================================

def train_neural_network(model, train_dataset, validation_dataset=None, epochs=10, callbacks=None, verbose=1):
    if not isinstance(model, tf.keras.Model):
        raise TypeError("Modelul trebuie să fie o instanță tf.keras.Model")

    if not model.compiled:
        raise ValueError("Modelul trebuie să fie compilat înainte de antrenare.")

    if callbacks is None:
        callbacks = []

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )
    return {
        'history': history.history,
        'epochs_completed': len(history.history['loss'])
    }

# ============================================================================
# 3. GESTIONARE PONDERI MODEL
# ============================================================================

def get_model_weights(model): return model.get_weights()
def set_model_weights(model, weights): model.set_weights(weights)

# ============================================================================
# 4. METRICI DE PERFORMANȚĂ
# ============================================================================

def calculate_metrics(model, test_dataset, average='weighted'):
    y_true, y_pred = [], []

    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
    }
    return metrics

# ============================================================================
# 5. SALVARE / ÎNCĂRCARE MODEL
# ============================================================================

def save_model_config(model, filepath, save_weights=True):
    if not filepath.endswith('.keras'):
        filepath += '.keras'
    model.save(filepath)
    print(f"Model salvat în: {filepath}")

def load_model_config(filepath):
    if not filepath.endswith('.keras'):
        filepath += '.keras'
    model = tf.keras.models.load_model(filepath)
    print(f"Model încărcat din: {filepath}")
    return model

# ============================================================================
# 6. FUNCȚII AUXILIARE
# ============================================================================

def save_weights_only(model, filepath): model.save_weights(filepath)
def load_weights_only(model, filepath): model.load_weights(filepath)

def validate_model_structure(model):
    info = {
        'total_params': model.count_params(),
        'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
        'layers_count': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'is_compiled': model.compiled
    }
    return info

# ============================================================================
# EXEMPLU DE UTILIZARE
# ============================================================================

def _model_compile(model):
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_model() -> tf.keras.Model:
    """CNN pentru clasificarea CIFAR-10"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ], name="CIFAR10_CNN")

    return _model_compile(model)

if __name__ == "__main__":
    train_ds, test_ds = load_train_test_data()
    train_ds, test_ds = preprocess_loaded_data(train_ds, test_ds)

    model = create_model()
    print(validate_model_structure(model))

    history = train_neural_network(model, train_ds, test_ds, epochs=5)
    metrics = calculate_metrics(model, test_ds)

    print("Rezultate:", metrics)
    save_model_config(model, model.name)
