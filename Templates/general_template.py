"""
Template General pentru Antrenarea Modelelor de Învățare Automată în Federated Learning
Compatibil cu Rețele Neuronale (TensorFlow/Keras) și Modele ML Pre-Antrenate

Acest template oferă o structură modulară pentru a permite utilizatorilor să antreneze modele în mod flexibil.
Poate fi adaptat pentru:
- Rețele neuronale simple (ex: pe MNIST sau date similare)
- Modele cu date descărcate de la URL-uri publice (ex: imagini de flori)
- Modele pre-antrenate (ex: MobileNetV2 pe CIFAR-10 sau date dummy)

Instrucțiuni pentru utilizator (prezentate în frontend):
1. Importați funcțiile necesare în scriptul dvs. principal.
2. Definiți funcția de încărcare a datelor (alegeți metoda potrivită: locală, de la URL, dummy sau reală).
3. Creați și compilați modelul (simplu sau pre-antrenat).
4. Preprocesați datele.
5. Antrenați modelul.
6. Extrageți ponderi, calculați metrici și salvați modelul.
7. Adaptați secțiunea __main__ pentru testare.

Notă: Acest template integrează elemente din template_code.py, template_code3.py și template_code5.py.
Asigurați-vă că aveți TensorFlow instalat (pip install tensorflow).

Actualizare: Funcția pentru încărcarea CIFAR-10 a fost optimizată pentru a evita OOM prin preprocesare lazy cu .map().
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any, Optional
import json
import requests
import os
import shutil
import tarfile

# ============================================================================
# 0. CONFIGURARE GENERALĂ (Adaptați după nevoie)
# ============================================================================
# Configurații implicite - pot fi suprascrise de utilizator
DEFAULT_IMG_SIZE = (224, 224)  # Dimensiune standard pentru modele pre-antrenate
DEFAULT_NUM_CLASSES = 10  # Număr clase implicite (ex: MNIST sau CIFAR-10)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 5
DEFAULT_MODEL_NAME = "CustomModel"
DEFAULT_WEIGHTS = 'imagenet'  # Pentru modele pre-antrenate
PUBLIC_DATA_URL = None  # Setați un URL dacă descărcați date publice (ex: 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz')
DATA_DIR = './data'  # Director temporar pentru date

# ============================================================================
# 1. FUNCȚII PENTRU ÎNCĂRCAREA ȘI DESCĂRCAREA DATELOR
# ============================================================================

def download_public_data(url: str, save_path: str) -> bool:
    """Descarcă un fișier de la un URL public (din template_code3.py)."""
    print(f"Începe descărcarea de la: {url}")
    try:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Descărcare reușită: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Eroare descărcare: {e}")
        return False

def extract_archive(file_path: str, extract_dir: str) -> None:
    """Dezarhivează un fișier tar.gz (din template_code3.py)."""
    if file_path.endswith(('.tar.gz', '.tgz')):
        print(f"Dezarhivare {file_path} în {extract_dir}...")
        try:
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
            print("✅ Dezarhivare reușită!")
        except Exception as e:
            print(f"❌ Eroare dezarhivare: {e}")

def load_train_test_data(
    source: str = 'local',  # Opțiuni: 'local' (ex: MNIST), 'public_url', 'dummy', 'real_url'
    url: Optional[str] = None,
    num_samples: int = 1000,
    num_classes: int = DEFAULT_NUM_CLASSES,
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Funcție generală pentru încărcarea datelor (integrează din toate template-urile).
    
    Args:
        source: Tipul sursei ('local', 'public_url', 'dummy', 'real_url')
        url: URL pentru descărcare (dacă source='public_url' sau 'real_url')
        num_samples: Pentru dummy dataset
        num_classes: Număr clase
        img_size: Dimensiune imagini
    
    Returns:
        (train_dataset, test_dataset)
    """
    if source == 'local':
        # Exemplu MNIST (din template_code.py)
        import tensorflow_datasets as tfds
        ds_train = tfds.load('mnist', split='train[:80%]', as_supervised=True)
        ds_test = tfds.load('mnist', split='train[80%:]', as_supervised=True)
        print("✅ Date locale (MNIST) încărcate!")
        return ds_train, ds_test
    
    elif source == 'public_url':
        # Descărcare de la URL public (din template_code3.py)
        if not url:
            raise ValueError("URL necesar pentru source='public_url'")
        full_path = os.path.join(DATA_DIR, url.split('/')[-1])
        if not download_public_data(url, full_path):
            raise ConnectionError("Eșec descărcare date.")
        extract_dir = os.path.join(DATA_DIR, 'extracted')
        extract_archive(full_path, extract_dir)
        base_dir = os.path.join(extract_dir, 'flower_photos')  # Adaptați după arhivă
        train_ds = tf.keras.utils.image_dataset_from_directory(base_dir, validation_split=0.2, subset="training", seed=123, image_size=img_size, batch_size=DEFAULT_BATCH_SIZE)
        test_ds = tf.keras.utils.image_dataset_from_directory(base_dir, validation_split=0.2, subset="validation", seed=123, image_size=img_size, batch_size=DEFAULT_BATCH_SIZE)
        print("✅ Date publice descărcate și încărcate!")
        return train_ds, test_ds
    
    elif source == 'dummy':
        # Dataset dummy (din template_code5.py)
        train_size = int(num_samples * 0.8)
        train_images = np.random.rand(train_size, *img_size, 3).astype(np.float32)
        train_labels = np.random.randint(0, num_classes, train_size).astype(np.int32)
        test_images = np.random.rand(num_samples - train_size, *img_size, 3).astype(np.float32)
        test_labels = np.random.randint(0, num_classes, num_samples - train_size).astype(np.int32)
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        print("✅ Dataset dummy creat!")
        return train_ds, test_ds
    
    elif source == 'real_url':
        # Exemplu CIFAR-10 optimizat pentru OOM (din template_code5.py, cu preprocesare lazy)
        print("\n📥 Încărcare dataset real: CIFAR-10")
        try:
            # Încărcăm CIFAR-10 fără redimensionare imediată
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            
            # Flatten labels (pentru sparse categorical crossentropy)
            y_train = y_train.flatten()
            y_test = y_test.flatten()
            
            # Creăm tf.data.Dataset
            train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            
            # Definim funcția de preprocesare (resize + normalizare per imagine)
            def preprocess_fn(image, label):
                image = tf.cast(image, tf.float32)  # Asigură tipul corect
                image = tf.image.resize(image, img_size)  # Resize per imagine
                image = image / 255.0  # Normalizare
                return image, label
            
            # Aplicăm preprocesarea lazy cu .map() - folosește paralelism pentru eficiență
            train_ds = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
            test_ds = test_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
            
            print(f"✅ Dataset încărcat cu preprocesare lazy!")
            print(f"   Train samples: {len(x_train)}")
            print(f"   Test samples: {len(x_test)}")
            print(f"   Classes: 10 (CIFAR-10)")
            
            return train_ds, test_ds
            
        except Exception as e:
            print(f"⚠️ Nu s-a putut încărca CIFAR-10: {e}")
            print("   Se creează dataset dummy...")
            # Fallback la dummy
            train_size = int(num_samples * 0.8)
            train_images = np.random.rand(train_size, *img_size, 3).astype(np.float32)
            train_labels = np.random.randint(0, num_classes, train_size).astype(np.int32)
            test_images = np.random.rand(num_samples - train_size, *img_size, 3).astype(np.float32)
            test_labels = np.random.randint(0, num_classes, num_samples - train_size).astype(np.int32)
            train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            print("✅ Dataset dummy creat ca fallback!")
            return train_ds, test_ds
    
    else:
        raise ValueError(f"Sursă invalidă: {source}. Opțiuni: 'local', 'public_url', 'dummy', 'real_url'")

def preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Preprocesare simplă: normalizare și one-hot (din template_code.py și template_code3.py)."""
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, DEFAULT_NUM_CLASSES) if 'categorical' in get_loss_type() else label
    return image, label

def preprocess_loaded_data(
    train_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle_buffer: int = 1000
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Aplică preprocesare, shuffle și batch (din toate template-urile)."""
    # Notă: Pentru 'real_url', preprocesarea este deja aplicată în load_train_test_data, deci aici o aplicăm doar dacă este necesar
    train_ds = train_ds.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds

# ============================================================================
# 2. FUNCȚII PENTRU CREAREA ȘI COMPILAREA MODELULUI
# ============================================================================

def create_model(
    model_type: str = 'simple',  # Opțiuni: 'simple' (CNN), 'pretrained' (MobileNetV2 etc.)
    num_classes: int = DEFAULT_NUM_CLASSES,
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
    pretrained_name: str = 'MobileNetV2',
    weights: str = DEFAULT_WEIGHTS
) -> tf.keras.Model:
    """
    Creează modelul (integrează din template_code.py, template_code3.py, template_code5.py).
    
    Args:
        model_type: 'simple' sau 'pretrained'
        num_classes: Număr clase
        img_size: Dimensiune input
        pretrained_name: Nume model pre-antrenat (ex: 'MobileNetV2', 'ResNet50')
        weights: 'imagenet' sau None
    
    Returns:
        Model Keras necompilat
    """
    if model_type == 'simple':
        # Model simplu CNN (din template_code3.py sau template_code.py)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(*img_size, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ], name="Simple_CNN_Model")
        print("✅ Model simplu creat!")
    
    elif model_type == 'pretrained':
        # Model pre-antrenat (din template_code5.py)
        models_dict = {
            'MobileNetV2': tf.keras.applications.MobileNetV2,
            'ResNet50': tf.keras.applications.ResNet50,
            # Adăugați altele după nevoie
        }
        if pretrained_name not in models_dict:
            raise ValueError(f"Model pre-antrenat invalid: {pretrained_name}")
        base_model = models_dict[pretrained_name](weights=weights, include_top=False, input_shape=(*img_size, 3))
        base_model.trainable = False
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ], name=f"{pretrained_name}_Pretrained")
        print("✅ Model pre-antrenat creat!")
    
    else:
        raise ValueError(f"Tip model invalid: {model_type}. Opțiuni: 'simple', 'pretrained'")
    
    return model

def compile_model(
    model: tf.keras.Model,
    optimizer: str = 'adam',
    learning_rate: float = 1e-3,
    loss: str = 'categorical_crossentropy',
    metrics: list = ['accuracy']
) -> tf.keras.Model:
    """Compilează modelul (din template_code5.py)."""
    opt = tf.keras.optimizers.Adam(learning_rate) if optimizer == 'adam' else tf.keras.optimizers.SGD(learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    print("✅ Model compilat!")
    return model

# ============================================================================
# 3. FUNCȚIE PENTRU ANTRENAREA MODELULUI (din template_code.py)
# ============================================================================

def train_neural_network(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset = None,
    epochs: int = DEFAULT_EPOCHS,
    callbacks: list = None,
    verbose: int = 1
) -> Dict[str, Any]:
    if callbacks is None:
        callbacks = []
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=callbacks, verbose=verbose)
    return {'history': history.history, 'epochs_completed': len(history.history['loss'])}

# ============================================================================
# 4. FUNCȚII PENTRU PONDERI, METRICI, VALIDARE (din template-urile originale)
# ============================================================================

def get_model_weights(model: tf.keras.Model) -> list:
    return model.get_weights()

def set_model_weights(model: tf.keras.Model, weights: list) -> None:
    model.set_weights(weights)

def calculate_metrics(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    average: str = 'weighted'
) -> Dict[str, float]:
    y_true, y_pred = [], []
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1) if predictions.shape[1] > 1 else (predictions > 0.5).astype(int).flatten())
        y_true.extend(np.argmax(labels.numpy(), axis=1) if labels.shape[1] > 1 else labels.numpy().flatten())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
    }

def validate_model_structure(model: tf.keras.Model) -> Dict[str, Any]:
    is_compiled = True if hasattr(model, 'optimizer') else False
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
        info['loss'] = str(model.loss)
    return info

# ============================================================================
# 5. FUNCȚII AUXILIARE (Salvare, Fine-Tuning etc.)
# ============================================================================

def save_model_config(model: tf.keras.Model, filepath: str) -> None:
    model.save(filepath)
    print(f"Model salvat la: {filepath}")

def unfreeze_model_layers(model: tf.keras.Model, num_layers: int = 20) -> tf.keras.Model:
    for layer in model.layers[-num_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    return model

def get_loss_type() -> str:
    return 'categorical_crossentropy'  # Adaptați după nevoie

# ============================================================================
# EXEMPLU DE UTILIZARE (Adaptați în scriptul dvs. principal sau Colab)
# ============================================================================

if __name__ == "__main__":
    # Exemplu 1: Model simplu pe date locale (MNIST)
    # train_ds, test_ds = load_train_test_data(source='local')
    # train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    # test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    # train_ds, test_ds = preprocess_loaded_data(train_ds, test_ds)
    # model = create_model(model_type='simple', num_classes=10, img_size=(28, 28))
    # model = compile_model(model, loss='categorical_crossentropy')
    
    # Exemplu 2: Model pre-antrenat pe date reale (CIFAR-10, cu fix OOM)
    train_ds, test_ds = load_train_test_data(source='real_url', img_size=DEFAULT_IMG_SIZE)
    train_ds, test_ds = preprocess_loaded_data(train_ds, test_ds)
    model = create_model(model_type='pretrained', num_classes=10, pretrained_name='MobileNetV2')
    model = compile_model(model, loss='sparse_categorical_crossentropy')  # Sparse pentru labels integer
    
    # Validare
    print(validate_model_structure(model))
    
    # Antrenare
    history = train_neural_network(model, train_ds, test_ds, epochs=3)
    
    # Metrici
    metrics = calculate_metrics(model, test_ds)
    print(metrics)
    
    # Salvare
    save_model_config(model, 'my_model.keras')
    
    # Curățare (opțional)
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)