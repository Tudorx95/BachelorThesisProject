import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any
import json
import requests
import os
import shutil
import tarfile

# ============================================================================
# 0. CONFIGURARE URL PUBLIC & CĂI LOCALE
# ============================================================================
PUBLIC_URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
DATA_DIR = 'downloaded_image_data'
FILE_NAME = PUBLIC_URL.split('/')[-1]

# ============================================================================
# 1. FUNCȚIE PENTRU EXTRAGEREA DATELOR
# ============================================================================
def download_public_data(url: str, save_path: str) -> bool:
    """Descarcă un fișier de la un URL public."""
    print(f"Începe testarea conectivității și descărcarea de la: {url}")
    try:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status() 
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Descărcare reușită! Fișier salvat la: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ EROARE de conectivitate/descărcare: {e}")
        return False

def load_train_test_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Descarcă datele, le dezarhivează și le încarcă în format tf.data.Dataset.
    
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: (train_dataset, test_dataset)
    """
    full_path = os.path.join(DATA_DIR, FILE_NAME)
    
    # Pasul A: Testarea conectivității și descărcare
    if not download_public_data(PUBLIC_URL, full_path):
        raise ConnectionError("Nu s-a putut descărca fișierul. Conexiunea la internet a eșuat.")
    
    # Pasul B: Dezarhivare
    extract_dir = os.path.join(DATA_DIR, 'extracted')
    if full_path.endswith(('.tar.gz', '.tgz')):
        print(f"Dezarhivare {FILE_NAME} în {extract_dir}...")
        try:
            with tarfile.open(full_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
        except Exception as e:
            print(f"Eroare la dezarhivare: {e}")
    
    # Pasul C: Încărcarea imaginilor
    try:
        print("Încărcare imagini din foldere...")
        base_dir = os.path.join(extract_dir, 'flower_photos')
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
            base_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(180, 180),
            batch_size=32
        )
        test_ds = tf.keras.utils.image_dataset_from_directory(
            base_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(180, 180),
            batch_size=32
        )
        return train_ds, test_ds
        
    except Exception as e:
        print(f"Eroare la încărcarea datelor: {e}. Returnare seturi dummy.")
        dummy_images = tf.zeros((32, 180, 180, 3), dtype=tf.float32)
        dummy_labels = tf.one_hot(tf.zeros(32, dtype=tf.int32), depth=5)
        dummy_ds = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels)).batch(32)
        return dummy_ds, dummy_ds

def preprocess(image, label):
    """Normalizare simplă (0-1) pentru imagini."""
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def preprocess_loaded_data(train_ds, test_ds) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Aplică funcția de preprocesare și configurează prefetching."""
    train_ds_processed = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    test_ds_processed = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    return train_ds_processed, test_ds_processed

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
    """
    if not isinstance(model, tf.keras.Model):
        raise TypeError("Modelul trebuie să fie o instanță tf.keras.Model")
    
    # CORECȚIE: Verificare mai robustă pentru compilare
    try:
        # Încercăm să accesăm optimizer-ul - va eșua dacă modelul nu e compilat
        _ = model.optimizer
        is_compiled = True
    except (AttributeError, ValueError):
        is_compiled = False
    
    if not is_compiled:
        raise ValueError(
            "Modelul trebuie să fie compilat înainte de antrenare. "
            "Folosiți model.compile(optimizer=..., loss=..., metrics=...)"
        )
    
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
# 3. FUNCȚII PENTRU EXTRAGEREA/SETAREA PONDERILOR
# ============================================================================
def get_model_weights(model: tf.keras.Model) -> list:
    """Extrage ponderile (weights) din model."""
    return model.get_weights()

def set_model_weights(model: tf.keras.Model, weights: list) -> None:
    """Setează ponderile (weights) în model."""
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
    """
    y_true = []
    y_pred = []
    
    for batch_data in test_dataset:
        if len(batch_data) == 2:
            images, labels = batch_data
        else:
            raise ValueError("Dataset-ul trebuie să conțină (features, labels)")
        
        predictions = model.predict(images, verbose=0)
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            y_pred.extend(np.argmax(predictions, axis=1))
        else:
            y_pred.extend((predictions > 0.5).astype(int).flatten())
        
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        else:
            y_true.extend(labels.numpy().flatten())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
    }
    
    return metrics

# ============================================================================
# 5. FUNCȚIE DE VALIDARE A MODELULUI
# ============================================================================
def validate_model_structure(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Validează și returnează informații despre structura modelului.
    """
    # CORECȚIE: Verificare mai robustă
    try:
        _ = model.optimizer
        is_compiled = True
    except (AttributeError, ValueError):
        is_compiled = False
    
    info = {
        'total_params': model.count_params(),
        'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
        'layers_count': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'is_compiled': is_compiled
    }
   
    if is_compiled:
        info['optimizer'] = model.optimizer.__class__.__name__
        info['loss'] = model.loss.__class__.__name__ if hasattr(model.loss, '__class__') else str(model.loss)
    
    return info

# ============================================================================
# FUNCȚII AUXILIARE PENTRU MODEL ȘI COMPILE
# ============================================================================
def create_model(num_classes=5):
    """Exemplu de model simplu CNN pentru imaginile RGB (180x180x3)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(180, 180, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name="PublicData_CNN_Model")
    
    # CORECȚIE: Compilăm modelul imediat după creare
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# EXEMPLU DE UTILIZARE
# ============================================================================
if __name__ == "__main__":
    
    print("--- Test de Conectivitate și Antrenare ---")
    
    try:
        # Pasul 1: Încărcare date
        train_ds, test_ds = load_train_test_data()
        
        # Pasul 2: Preprocesare
        train_ds, test_ds = preprocess_loaded_data(train_ds, test_ds)
        
        # Pasul 3: Creare model (DEJA COMPILAT)
        model = create_model()
        
        # Pasul 4: Validare structură
        print("\n--- Informații Model ---")
        model_info = validate_model_structure(model)
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # Pasul 5: Antrenare
        print("\n--- Începe antrenarea ---")
        history = train_neural_network(
            model=model,
            train_dataset=train_ds,
            validation_dataset=test_ds,
            epochs=2,  # Mărită la 2 epoci pentru test mai bun
            verbose=1
        )
        
        print(f"\n✅ Antrenare finalizată după {history['epochs_completed']} epoci")
        
        # Pasul 6: Extragere ponderi
        weights = get_model_weights(model)
        print(f"   Număr de layere cu ponderi extrase: {len(weights)}")
        
        # Pasul 7: Calculare metrici
        print("\n--- Calculare metrici ---")
        metrics = calculate_metrics(model, test_ds, average='macro')
        print("--- Metrici după antrenare ---")
        for metric_name, value in metrics.items():
            print(f"   {metric_name}: {value:.4f}")
        
        # Pasul 8: Salvare model (opțional)
        model_path = "trained_flower_model.h5"
        model.save(model_path)
        print(f"\n✅ Model salvat la: {model_path}")
            
    except Exception as e:
        import traceback
        print(f"\n❌ Eșec în execuția completă. Eroare: {e}")
        print("\nTraceback complet:")
        traceback.print_exc()
        
    finally:
        # Curățare fișiere descărcate
        if os.path.exists(DATA_DIR):
            print(f"\nCurățare director temporar: {DATA_DIR}")
            shutil.rmtree(DATA_DIR)