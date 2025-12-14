"""
Template pentru antrenarea rețelelor neuronale în medii Federated Learning
Compatibil cu orice arhitectură TensorFlow/Keras
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any
import json


# ============================================================================
# 1. FUNCȚIE PENTRU EXTRAGEREA DATELOR (completată de user în Colab)
# ============================================================================



def load_train_test_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Funcție care trebuie completată de utilizator pentru a încărca datele.
    
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: (train_dataset, test_dataset)
    """
    """Implementare exemplu pentru MNIST"""
    import tensorflow_datasets as tfds
    
    ds_train = tfds.load('mnist', split='train[:80%]', as_supervised=True)
    ds_test = tfds.load('mnist', split='train[80%:]', as_supervised=True)

    return ds_train, ds_test

def preprocess(image, label):
        """
        Preprocesare de bază pentru imagini și label-uri.
        Aceasta este funcția folosită în FL simulation.
        """
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, (28, 28))
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, 10)
        return image, label

def preprocess_loaded_data(train_ds, test_ds)-> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Preprocesează dataset-urile încărcate.
    
    Args:
        train_ds: Dataset de antrenare brut
        test_ds: Dataset de testare brut
    
    Returns:
        Tuple de dataset-uri preprocesate
    """
    train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds


def load_client_data(data_path: str, batch_size: int = 32) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Funcție pentru încărcarea datelor în FL simulator (AGNOSTIC).
    
    Această funcție este apelată de fd_simulator pentru fiecare client.
    Implementarea este COMPLETĂ - simulatorul NU aplică nicio preprocesare proprie!
    
    Args:
        data_path: Path către directorul cu date (ex: "clean_data" sau "clean_data_poisoned")
        batch_size: Dimensiunea batch-ului
        
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: (train_ds, test_ds) complet preprocesate
        
    Important:
        - Datele returnate TREBUIE să fie gata pentru antrenare (preprocessed, batched)
        - Simulatorul NU va aplica nicio transformare suplimentară
        - Această funcție oferă control TOTAL asupra preprocesării
    """
    from pathlib import Path
    
    data_path = Path(data_path)
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Data directories not found: {data_path}")
    
    # Încarcă datele din directoare
    # label_mode='int' pentru că vom aplica one-hot manual în preprocess
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(28, 28),
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='int',  # Labels ca indici întregi (0-9)
        shuffle=True,
        seed=42
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(28, 28),
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='int',
        shuffle=False
    )
    
    # Aplică preprocesare COMPLETĂ
    def preprocess_for_fl(image, label):
        """Preprocesare specifică FL - normalizare + one-hot encoding."""
        # Normalizare
        image = tf.cast(image, tf.float32) / 255.0
        
        # One-hot encoding
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, 10)
        
        return image, label
    
    train_ds = train_ds.map(preprocess_for_fl, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_for_fl, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Optimizare
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds


def download_data(output_dir: str = "clean_data"):
    """
    Descarcă, preprocesează și salvează datele în două directoare separate:
    clean_data/train/<clasă>/ și clean_data/test/<clasă>/.
    Această funcție este apelată de orchestrator.
    """
    import tensorflow as tf
    import numpy as np
    from pathlib import Path
    import os
    import json
    from PIL import Image

    output_path = Path(output_dir)
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading and saving data to {output_dir}...")

    # 1️⃣ Încarcă datele
    train_ds, test_ds = load_train_test_data()
    train_ds, test_ds = preprocess_loaded_data(train_ds, test_ds)

    # 2️⃣ Convertește în numpy
    X_train, y_train = [], []
    X_test, y_test = [], []

    for batch_images, batch_labels in train_ds:
        X_train.append(batch_images.numpy())
        y_train.append(batch_labels.numpy())

    for batch_images, batch_labels in test_ds:
        X_test.append(batch_images.numpy())
        y_test.append(batch_labels.numpy())

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Dacă etichetele sunt one-hot, convertim în indici
    if y_train.ndim > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    num_classes = len(np.unique(y_train))

    # 3️⃣ Salvează imaginile în directoare
    def save_images(X, y, base_dir):
        for i, (img_array, label) in enumerate(zip(X, y)):
            class_dir = base_dir / str(label)
            class_dir.mkdir(parents=True, exist_ok=True)

            # Convertește tensorul în imagine PIL
            img = (img_array * 255).astype(np.uint8)
            if img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
            img_pil = Image.fromarray(img)
            img_pil.save(class_dir / f"{i:05d}.jpg")

    print("Saving training images...")
    save_images(X_train, y_train, train_dir)
    print("Saving test images...")
    save_images(X_test, y_test, test_dir)

    # 4️⃣ Creează metadata
    metadata = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "input_shape": list(X_train[0].shape),
        "num_classes": int(num_classes)
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Data saved successfully!")
    print(f"  - Train samples: {metadata['train_samples']}")
    print(f"  - Test samples: {metadata['test_samples']}")
    print(f"  - Input shape: {metadata['input_shape']}")
    print(f"  - Number of classes: {metadata['num_classes']}")

    return metadata



# ============================================================================
# 2. FUNCȚIE PENTRU ANTRENAREA REȚELEI NEURONALE
# ============================================================================

def train_neural_network(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset = None,
    epochs: int = 10,
    callbacks: list = None,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Antrenează o rețea neuronală pe un dataset furnizat.
    Funcție generică compatibilă cu orice arhitectură Keras.
    
    Args:
        model: Model Keras pre-construit și compilat
        train_dataset: Dataset de antrenare (tf.data.Dataset)
        validation_dataset: Dataset de validare (opțional)
        epochs: Număr de epoci
        callbacks: Lista de callbacks Keras (opțional)
        verbose: Nivel de detaliere (0, 1, sau 2)
    
    Returns:
        Dict cu istoricul antrenării
    """
    if not isinstance(model, tf.keras.Model):
        raise TypeError("Modelul trebuie să fie o instanță tf.keras.Model")
    
    # Verifică dacă modelul este compilat
    if not hasattr(model, 'optimizer') or model.optimizer is None:
        raise ValueError(
            "Modelul trebuie să fie compilat înainte de antrenare. "
            "Folosiți model.compile(optimizer=..., loss=..., metrics=...)"
        )
    
    if callbacks is None:
        callbacks = []
    
    # Antrenare
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
# 3. FUNCȚIE PENTRU EXTRAGEREA PONDERILOR (WEIGHTS)
# ============================================================================

def get_model_weights(model: tf.keras.Model) -> list:
    """
    Extrage ponderile (weights) din model.
    Util pentru agregarea în Federated Learning.
    
    Args:
        model: Model Keras
    
    Returns:
        Lista de array-uri NumPy cu ponderile modelului
    """
    return model.get_weights()


def set_model_weights(model: tf.keras.Model, weights: list) -> None:
    """
    Setează ponderile (weights) în model.
    Util pentru aplicarea ponderilor agregate în Federated Learning.
    
    Args:
        model: Model Keras
        weights: Lista de array-uri NumPy cu ponderile
    """
    model.set_weights(weights)


# ============================================================================
# 4. FUNCȚIE PENTRU CALCULAREA METRICILOR
# ============================================================================

def calculate_metrics(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculează metrici de evaluare: accuracy, precision, recall, f1_score.
    
    Args:
        model: Model Keras antrenat
        test_dataset: Dataset de test
        average: Tip de medie pentru metrici ('weighted', 'macro', 'micro')
    
    Returns:
        Dict cu metrici: {'accuracy', 'precision', 'recall', 'f1_score'}
    """
    y_true = []
    y_pred = []
    
    for batch_data in test_dataset:
        # daca batch-ul e in formatul (feature, label) => extragem pe fiecare 
        if len(batch_data) == 2:
            images, labels = batch_data
        else:
            raise ValueError("Dataset-ul trebuie să conțină (features, labels)")
        
        # Genereaza predictii in functie de batch-ul curent 
        predictions = model.predict(images, verbose=0)
        # daca e predictie binara => vector de probabilitati 
        # daca e predictie multi-class => matrice [num_samples (adica imagini), num_classes (adica etichete/labels)]
        # ex. labels 0,1,2 si 4 imagini => matrice (4,3)
        
        # predictions.shape => ne arata dim_batch (cate imagini) si pt fiecare o lista de dim=num_clase cu probabilitatile de apartenenta la fiecare

        # Convertire la etichete
        # prediction.shape = (batch_size, num_clase)
        # daca avem mai multe etichete, alege pe cea cu probabilitatea cea mai mare 
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Clasificare multi-clasă
            y_pred.extend(np.argmax(predictions, axis=1))
        else:
            # Clasificare binară
            y_pred.extend((predictions > 0.5).astype(int).flatten()) # pune 1 daca probabilitatea > 0.5, altfel pune 0
        
        # Convertire labels la format corect (scalar)
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            # One-hot encoded (ex. [0, 0, 1] => clasa 2 adica transforma in index de label)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        else:
            # Deja în format scalar 
            y_true.extend(labels.numpy().flatten())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculare metrici
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
    }
    
    return metrics


# ============================================================================
# 5. FUNCȚII PENTRU SALVARE/ÎNCĂRCARE CONFIGURAȚIE MODEL
# ============================================================================




def save_model_config(
    model: tf.keras.Model,
    filepath: str,
    save_weights: bool = True
) -> None:
    """
    Salvează configurația completă a modelului în format .keras.
    
    Args:
        model: Model Keras
        filepath: Calea către fișierul de salvare (ex: 'model.keras')
        save_weights: Dacă True, salvează și ponderile
    """
    if not filepath.endswith('.keras'):
        filepath += '.keras'
    
    if save_weights:
        # Salvează model complet (arhitectură + ponderi + optimizer state)
        model.save(filepath)
    else:
        # Salvează doar arhitectura
        model_json = model.to_json()
        config = {
            'architecture': model_json,
            'config': model.get_config()
        }
        
        with open(filepath.replace('.keras', '_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"Model salvat în: {filepath}")


def load_model_config(filepath: str) -> tf.keras.Model:
    """
    Încarcă configurația modelului din fișier .keras.
    
    Args:
        filepath: Calea către fișierul salvat
    
    Returns:
        Model Keras încărcat
    """
    if not filepath.endswith('.keras'):
        filepath += '.keras'
    
    try:
        # Încarcă model complet
        model = tf.keras.models.load_model(filepath)
        print(f"Model încărcat din: {filepath}")
        return model
    except Exception as e:
        print(f"Eroare la încărcarea modelului: {e}")
        raise


# ============================================================================
# 6. FUNCȚIE AUXILIARĂ PENTRU SALVAREA PONDERILOR SEPARATE
# ============================================================================

def save_weights_only(model: tf.keras.Model, filepath: str) -> None:
    """
    Salvează doar ponderile modelului (fără arhitectură).
    Util pentru transferul rapid de ponderi în Federated Learning.
    
    Args:
        model: Model Keras
        filepath: Calea către fișier (ex: 'weights.h5')
    """
    model.save_weights(filepath)
    print(f"Ponderi salvate în: {filepath}")


def load_weights_only(model: tf.keras.Model, filepath: str) -> None:
    """
    Încarcă doar ponderile în model (arhitectura trebuie să existe deja).
    
    Args:
        model: Model Keras cu arhitectura corectă
        filepath: Calea către fișierul cu ponderi
    """
    model.load_weights(filepath)
    print(f"Ponderi încărcate din: {filepath}")


# ============================================================================
# 7. FUNCȚIE DE VALIDARE A MODELULUI
# ============================================================================

def validate_model_structure(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Validează și returnează informații despre structura modelului.
    Util pentru verificarea compatibilității înainte de antrenare.
    
    Args:
        model: Model Keras
    
    Returns:
        Dict cu informații despre model
    """
    info = {
        'total_params': model.count_params(),
        'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
        'layers_count': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'is_compiled': hasattr(model, 'optimizer') and model.optimizer is not None
    }
    
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        info['optimizer'] = model.optimizer.__class__.__name__
        info['loss'] = model.loss.__class__.__name__ if hasattr(model.loss, '__class__') else str(model.loss)
    
    return info


# ============================================================================
# EXEMPLU DE UTILIZARE
# ============================================================================

def _model_compile(model: tf.keras.Model) -> tf.keras.Model:
    """
    Compilează modelul cu setările specificate de user.
    
    Args:
        model: Model Keras necompilat
        
    Returns:
        Model Keras compilat
    """
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_loss_type() -> str:
    """
    Returnează tipul funcției de loss folosită.
    Opțiuni: 'sparse_categorical_crossentropy' sau 'categorical_crossentropy'
    
    Returns:
        str: Tipul funcției de loss
    """
    return 'categorical_crossentropy'

def get_image_format() -> dict:
    """
    Returnează formatul imaginilor așteptat de model.
    
    Returns:
        dict: Dicționar cu 'channels' (1 sau 3) și 'size' (tuple cu height, width)
    """
    return {
        'channels': 1,  # 1 pentru grayscale, 3 pentru RGB
        'size': (28, 28)  # Dimensiunea imaginilor
    }

def get_data_preprocessing() -> callable:
    """
    Returnează funcția de preprocesare a datelor.
    
    Returns:
        callable: Funcția de preprocesare
    """
    return preprocess  # sau altă funcție definită de user

# Pasul 2: Utilizatorul creează modelul
def create_model():
        """Exemplu de model simplu pentru MNIST"""
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ], name="MyMNIST_model")
        
        _model_compile(model)
        
        return model


if __name__ == "__main__":
    """
    Exemplu complet de utilizare a template-ului.
    Acest cod trebuie adaptat de utilizator în Colab.
    """
    
    # Pasul 1: Utilizatorul definește funcția de încărcare date
    # (în acest exemplu folosim MNIST pentru demonstrație)
        
    
    
    # Pasul 3: Încărcare date
    train_ds, test_ds = load_train_test_data()
    train_ds, test_ds = preprocess_loaded_data(train_ds, test_ds)
    
    # Pasul 4: Creare model
    model = create_model()
    
    # Pasul 5: Validare structură
    model_info = validate_model_structure(model)
    # print model features
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Pasul 6: Antrenare
    history = train_neural_network(
        model=model,
        train_dataset=train_ds,
        validation_dataset=test_ds,
        epochs=3,
        verbose=1
    )
    
    # Pasul 7: Extragere ponderi
    weights = get_model_weights(model)
    print(f"   Număr de layere cu ponderi: {len(weights)}")
    
    # Pasul 8: Calculare metrici
    metrics = calculate_metrics(model, test_ds)
    # print metrics 
    with open("init-metrics.txt", "a") as f:
        for metric_name, value in metrics.items():
            f.write(f"   {metric_name}: {value:.4f}\n")
    
    # Pasul 9: Salvare model
    filepath= f"{model.name}.keras"
    save_model_config(model, filepath)
    
    # Pasul 10: Încărcare model
    # loaded_model = load_model_config(filepath)
    
    print("Finish")