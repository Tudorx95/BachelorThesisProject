"""
Template pentru antrenarea rețelelor neuronale în medii Federated Learning
Compatibil cu orice arhitectură TensorFlow/Keras

Model: ResNet18 PRE-ANTRENAT descărcat de pe HuggingFace Hub
Dataset: CIFAR-10 (60,000 imagini 32x32 RGB, 10 clase)
Ideal pentru: testarea atacurilor de tip data poisoning și backdoor attacks

IMPORTANT: Acest script DESCARCĂ modelul pre-antrenat (nu-l creează de la 0)
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any
import json
import os


# ============================================================================
# CONFIGURAȚIE GLOBALĂ - CIFAR-10
# ============================================================================

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
NUM_CLASSES = 10
IMG_SIZE = (32, 32)  # CIFAR-10 native resolution

# CONFIGURARE HUGGINGFACE
HUGGINGFACE_REPO_ID = "Tudorx95/resnet18-cifar10"  # ✏️ ÎNLOCUIEȘTE CU REPO-UL TĂU
MODEL_FILENAME = "ResNet18_CIFAR10.keras"  # ✏️ ÎNLOCUIEȘTE CU NUMELE FIȘIERULUI TĂU


# ============================================================================
# 1. FUNCȚIE PENTRU EXTRAGEREA DATELOR (CIFAR-10)
# ============================================================================

def load_train_test_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Încarcă CIFAR-10 dataset folosind tf.keras.datasets.
    
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: (train_dataset, test_dataset)
    """
    print("\n📦 Descărcare CIFAR-10 dataset...")
    
    # Descarcă CIFAR-10 (descărcare automată la prima rulare)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    print(f"   ✓ Train: {len(x_train)} imagini")
    print(f"   ✓ Test: {len(x_test)} imagini")
    
    # Convertește în tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    return train_ds, test_ds


def preprocess(image, label):
    """
    Preprocesare de bază pentru imagini și label-uri CIFAR-10.
    Aceasta este funcția folosită în FL simulation.
    """
    # Normalizare [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, (*IMG_SIZE, 3))
    
    # Flatten label dacă e 2D
    if len(label.shape) > 0:
        label = tf.squeeze(label)
    
    # One-hot encoding
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)
    
    return image, label


def preprocess_loaded_data(train_ds, test_ds) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
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
        data_path: Path către directorul cu date
        batch_size: Dimensiunea batch-ului
        
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: (train_ds, test_ds) complet preprocesate
    """
    from pathlib import Path
    
    data_path = Path(data_path)
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Data directories not found: {data_path}")
    
    # Încarcă datele din directoare
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        color_mode='rgb',
        label_mode='int',
        shuffle=True,
        seed=42
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        color_mode='rgb',
        label_mode='int',
        shuffle=False
    )
    
    # Aplică preprocesare COMPLETĂ
    def preprocess_for_fl(image, label):
        """Preprocesare specifică FL - normalizare + one-hot encoding."""
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, NUM_CLASSES)
        return image, label
    
    train_ds = train_ds.map(preprocess_for_fl, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_for_fl, num_parallel_calls=tf.data.AUTOTUNE)
    
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds


def download_data(output_dir: str = "cifar10_data"):
    """
    Descarcă, preprocesează și salvează datele CIFAR-10.
    Această funcție este apelată de orchestrator.
    """
    import numpy as np
    from pathlib import Path
    from PIL import Image

    output_path = Path(output_dir)
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading and saving data to {output_dir}...")

    train_ds, test_ds = load_train_test_data()
    train_ds, test_ds = preprocess_loaded_data(train_ds, test_ds)

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

    if y_train.ndim > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    def save_images(X, y, base_dir):
        for i, (img_array, label) in enumerate(zip(X, y)):
            class_name = CIFAR10_CLASSES[int(label)]
            class_dir = base_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            img = (img_array * 255).astype(np.uint8)
            img_pil = Image.fromarray(img)
            img_pil.save(class_dir / f"{i:05d}.jpg")

    print("Saving training images...")
    save_images(X_train, y_train, train_dir)
    print("Saving test images...")
    save_images(X_test, y_test, test_dir)

    metadata = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "input_shape": list(X_train[0].shape),
        "num_classes": int(NUM_CLASSES),
        "class_names": CIFAR10_CLASSES,
        "dataset": "CIFAR-10"
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Data saved successfully!")
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
    """
    if not isinstance(model, tf.keras.Model):
        raise TypeError("Modelul trebuie să fie o instanță tf.keras.Model")
    
    if not hasattr(model, 'optimizer') or model.optimizer is None:
        raise ValueError("Modelul trebuie să fie compilat înainte de antrenare.")
    
    if callbacks is None:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_dataset else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]
    
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return history.history


# ============================================================================
# 3. FUNCȚII PENTRU EXTRAGEREA PONDERILOR
# ============================================================================

def get_model_weights(model: tf.keras.Model):
    """Extrage ponderile modelului sub formă de listă de array-uri numpy."""
    return model.get_weights()


def set_model_weights(model: tf.keras.Model, weights) -> None:
    """Setează ponderile modelului din listă de array-uri numpy."""
    model.set_weights(weights)


# ============================================================================
# 4. FUNCȚII PENTRU CALCULAREA METRICILOR
# ============================================================================

def calculate_metrics(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    average: str = 'macro'
) -> Dict[str, float]:
    """Calculează metricile de evaluare pe dataset de test."""
    y_true_list = []
    y_pred_list = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(labels.numpy(), axis=1)
        y_true_list.extend(y_true)
        y_pred_list.extend(y_pred)
    
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
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
    """Salvează configurația completă a modelului în format .keras."""
    if not filepath.endswith('.keras'):
        filepath += '.keras'
    
    if save_weights:
        model.save(filepath)
    else:
        model_json = model.to_json()
        config = {'architecture': model_json, 'config': model.get_config()}
        with open(filepath.replace('.keras', '_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"Model salvat în: {filepath}")


def load_model_config(filepath: str) -> tf.keras.Model:
    """Încarcă configurația modelului din fișier .keras."""
    if not filepath.endswith('.keras'):
        filepath += '.keras'
    
    try:
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
    """Salvează doar ponderile modelului (fără arhitectură)."""
    model.save_weights(filepath)
    print(f"Ponderi salvate în: {filepath}")


def load_weights_only(model: tf.keras.Model, filepath: str) -> None:
    """Încarcă doar ponderile în model (arhitectura trebuie să existe deja)."""
    model.load_weights(filepath)
    print(f"Ponderi încărcate din: {filepath}")


# ============================================================================
# 7. FUNCȚIE DE VALIDARE A MODELULUI
# ============================================================================

def validate_model_structure(model: tf.keras.Model) -> Dict[str, Any]:
    """Validează și returnează informații despre structura modelului."""
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
# 8. FUNCȚII DE CONFIGURARE
# ============================================================================

def _model_compile(model: tf.keras.Model) -> tf.keras.Model:
    """Compilează modelul cu setările specificate de user."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_loss_type() -> str:
    """Returnează tipul funcției de loss folosite."""
    return 'categorical_crossentropy'


def get_image_format() -> dict:
    """Returnează formatul imaginilor așteptat de model."""
    return {'channels': 3, 'size': IMG_SIZE}


def get_data_preprocessing() -> callable:
    """Returnează funcția de preprocesare a datelor."""
    return preprocess


# ============================================================================
# 9. DESCĂRCARE MODEL PRE-ANTRENAT DE PE HUGGINGFACE
# ============================================================================

def create_model():
    """
    Descarcă model PRE-ANTRENAT de pe HuggingFace Hub.
    
    IMPORTANT: Acest script NU creează modelul de la 0, ci îl DESCARCĂ!
    
    Pentru a folosi această funcție:
    1. Încarcă modelul tău .keras pe HuggingFace Hub
    2. Actualizează variabilele globale:
       - HUGGINGFACE_REPO_ID = "your-username/your-repo"
       - MODEL_FILENAME = "your-model.keras"
    3. (Opțional) Setează token pentru repo-uri private:
       export HUGGING_FACE_HUB_TOKEN="hf_xxxxx"
    
    Returns:
        Model Keras descărcat și gata de folosit
    """
    from huggingface_hub import hf_hub_download
    
    print("\n🔽 Descărcare model PRE-ANTRENAT de pe HuggingFace...")
    print(f"   Repo: {HUGGINGFACE_REPO_ID}")
    print(f"   Fișier: {MODEL_FILENAME}")
    
    try:
        # Obține token din environment (opțional, pentru repo-uri private)
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        if token:
            print(f"   🔐 Token găsit: hf_{'*' * 10}{token[-4:]}")
        else:
            print("   ℹ️  Fără token (repo public)")
        
        # Descarcă modelul (cu cache local automat)
        print("   ⏳ Descărcare în curs...")
        
        model_path = hf_hub_download(
            repo_id=HUGGINGFACE_REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir="./models",  # Cache local pentru reutilizare
            token=token,
            force_download=False  # Folosește cache dacă există
        )
        
        print(f"   ✓ Descărcat în: {model_path}")
        
        # Încarcă modelul
        print("   📂 Încărcare model în memorie...")
        model = tf.keras.models.load_model(model_path)
        
        print(f"   ✓ Model încărcat cu succes!")
        print(f"   📊 Arhitectură: {model.name}")
        print(f"   📊 Parametri: {model.count_params():,}")
        print(f"   📊 Input shape: {model.input_shape}")
        print(f"   📊 Output shape: {model.output_shape}")
        
        # Verifică dacă modelul e compilat
        if not hasattr(model, 'optimizer') or model.optimizer is None:
            print("\n   ⚠️  Modelul nu este compilat. Compilare automată...")
            _model_compile(model)
            print("   ✓ Model compilat!")
        else:
            print(f"   ✓ Model deja compilat (optimizer: {model.optimizer.__class__.__name__})")
        
        return model
        
    except Exception as e:
        print(f"\n   ❌ Eroare la descărcare model: {e}")
        print("\n   💡 Verificări necesare:")
        print(f"      1. Repo există: https://huggingface.co/{HUGGINGFACE_REPO_ID}")
        print(f"      2. Fișier există: {MODEL_FILENAME}")
        print("      3. Internet connection")
        
        if "401" in str(e) or "403" in str(e):
            print("\n      4. Pentru repo privat, setează token:")
            print("         export HUGGING_FACE_HUB_TOKEN='hf_xxxxx'")
        
        print("\n   📝 Setup rapid:")
        print(f"""
# 1. Instalează dependențe
pip install huggingface-hub tensorflow

# 2. (Opțional) Login pentru repo privat
huggingface-cli login

# 3. Verifică repo
huggingface-cli repo-info {HUGGINGFACE_REPO_ID}

# 4. Actualizează constante în script:
HUGGINGFACE_REPO_ID = "your-username/your-repo"
MODEL_FILENAME = "your-model.keras"
""")
        raise


# ============================================================================
# EXEMPLU DE UTILIZARE - CONTINUARE ANTRENARE
# ============================================================================

if __name__ == "__main__":
    """
    Exemplu complet: Descarcă model pre-antrenat și continuă antrenarea.
    """
    
    print("=" * 70)
    print("CONTINUARE ANTRENARE RESNET18 PE CIFAR-10")
    print(f"Model sursă: HuggingFace Hub ({HUGGINGFACE_REPO_ID})")
    print("=" * 70)
    
    # Pasul 1: Încărcare CIFAR-10
    print("\n[Pasul 1] Încărcare CIFAR-10 dataset...")
    train_ds, test_ds = load_train_test_data()
    train_ds, test_ds = preprocess_loaded_data(train_ds, test_ds)
    
    # Pasul 2: Descărcare model pre-antrenat
    print("\n[Pasul 2] Descărcare model pre-antrenat de pe HuggingFace...")
    model = create_model()
    
    # Pasul 3: Validare structură
    print("\n[Pasul 3] Validare structură model:")
    model_info = validate_model_structure(model)
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Pasul 4: Evaluare ÎNAINTE de antrenare suplimentară
    print("\n[Pasul 4] Evaluare model ÎNAINTE de fine-tuning...")
    metrics_before = calculate_metrics(model, test_ds)
    print("   Metrici inițiale (model pre-antrenat):")
    for metric_name, value in metrics_before.items():
        print(f"      {metric_name}: {value:.4f}")
    
    # # Pasul 5: Continuare antrenare (fine-tuning)
    # print("\n[Pasul 5] Continuare antrenare (fine-tuning cu 5 epoci)...")
    # print("   ℹ️  Modelul este deja antrenat - facem doar ajustări fine")
    
    # # Opțional: Reduce learning rate pentru fine-tuning
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Learning rate mai mic
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy']
    # )
    
    # history = train_neural_network(
    #     model=model,
    #     train_dataset=train_ds,
    #     validation_dataset=test_ds,
    #     epochs=5,  # Puține epoci pentru fine-tuning
    #     verbose=1
    # )
    
    # # Pasul 6: Evaluare DUPĂ antrenare suplimentară
    # print("\n[Pasul 6] Evaluare model DUPĂ fine-tuning...")
    # metrics_after = calculate_metrics(model, test_ds)
    # print("   Metrici finale:")
    # for metric_name, value in metrics_after.items():
    #     print(f"      {metric_name}: {value:.4f}")
    
    # # Comparație
    # print("\n   📈 Îmbunătățire:")
    # for metric_name in metrics_before.keys():
    #     diff = metrics_after[metric_name] - metrics_before[metric_name]
    #     symbol = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
    #     print(f"      {metric_name}: {diff:+.4f} {symbol}")
    
    # # Pasul 7: Extragere ponderi
    # print("\n[Pasul 7] Extragere ponderi...")
    # weights = get_model_weights(model)
    # print(f"   Număr de layere cu ponderi: {len(weights)}")
    
    # # Pasul 8: Salvare model actualizat
    print("\n[Pasul 8] Salvare model actualizat...")
    filepath = f"{model.name}_.keras"
    save_model_config(model, filepath)
    
    # print("\n" + "=" * 70)
    # print("✓ FINE-TUNING FINALIZAT CU SUCCES!")
    # print("=" * 70)
    # print(f"\n📦 Model actualizat salvat: {filepath}")
    # print(f"📊 Parametri: {model.count_params():,}")
    
    # print("\n💡 Următorii pași:")
    # print("   1. (Opțional) Upload modelul actualizat pe HuggingFace:")
    # print(f"      python upload_to_huggingface.py {filepath} {HUGGINGFACE_REPO_ID}")
    # print("\n   2. Folosește modelul în simulator-ul FL")
    
    # print("\n🔬 Pentru teste de data poisoning:")
    # print("   • Modelul pre-antrenat este ideal ca baseline")
    # print("   • Testează robustețea la atacuri backdoor")
    # print("   • Framework: ART (Adversarial Robustness Toolbox)")
    
    # print("\nFinish")