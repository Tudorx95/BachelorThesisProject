import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any
import os
import shutil

# ============================================================================
# 0. CONFIGURARE - Folosim un model simplu și rapid
# ============================================================================
# Vom folosi MobileNetV2 pre-antrenat, disponibil direct în Keras
# Este echivalentul unui model de pe HuggingFace dar mai ușor de integrat
MODEL_NAME = "MobileNetV2"
NUM_CLASSES = 5  # Vom folosi 5 clase pentru exemplu
IMG_SIZE = (224, 224)  # Dimensiunea standard pentru MobileNetV2

# Alternative pentru încărcare directă din tf.keras.applications:
# - ResNet50, ResNet101, ResNet152
# - VGG16, VGG19
# - InceptionV3, InceptionResNetV2
# - DenseNet121, DenseNet169, DenseNet201
# - EfficientNetB0, EfficientNetB1, ... EfficientNetB7
# - NASNetMobile, NASNetLarge

DATA_DIR = './hf_data'

# ============================================================================
# 1. FUNCȚII PENTRU DESCĂRCAREA MODELULUI "HUGGINGFACE-STYLE"
# ============================================================================
def download_pretrained_model(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    weights: str = 'imagenet'
) -> tf.keras.Model:
    """
    Descarcă un model pre-antrenat (similar cu HuggingFace).
    Folosim tf.keras.applications care oferă modele pre-antrenate.
    
    Args:
        model_name: Numele modelului (MobileNetV2, ResNet50, etc.)
        num_classes: Numărul de clase pentru fine-tuning
        weights: 'imagenet' pentru ponderi pre-antrenate sau None
    
    Returns:
        Model TensorFlow/Keras
    """
    print(f"📥 Descărcare model pre-antrenat: {model_name}")
    print(f"   Ponderi inițiale: {weights}")
    print(f"   Număr clase finale: {num_classes}")
    
    try:
        # Dicționar cu modele disponibile
        models_dict = {
            'MobileNetV2': tf.keras.applications.MobileNetV2,
            'ResNet50': tf.keras.applications.ResNet50,
            'ResNet101': tf.keras.applications.ResNet101,
            'VGG16': tf.keras.applications.VGG16,
            'VGG19': tf.keras.applications.VGG19,
            'InceptionV3': tf.keras.applications.InceptionV3,
            'DenseNet121': tf.keras.applications.DenseNet121,
            'EfficientNetB0': tf.keras.applications.EfficientNetB0,
            'EfficientNetB1': tf.keras.applications.EfficientNetB1,
        }
        
        if model_name not in models_dict:
            raise ValueError(f"Model {model_name} nu este suportat. Opțiuni: {list(models_dict.keys())}")
        
        # Încărcăm modelul de bază fără top layer
        base_model = models_dict[model_name](
            weights=weights,
            include_top=False,
            input_shape=(*IMG_SIZE, 3),
            pooling='avg'  # Global average pooling
        )
        
        # Înghețăm layerele de bază (pentru fine-tuning mai rapid)
        base_model.trainable = False
        
        # Construim modelul complet cu layere custom
        inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name=f"{model_name}_FineTuned")
        
        print(f"✅ Model descărcat cu succes!")
        print(f"   Total parametri: {model.count_params():,}")
        print(f"   Parametri antrenabili: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Eroare la descărcarea modelului: {e}")
        raise

# ============================================================================
# 2. FUNCȚII PENTRU ÎNCĂRCAREA DATELOR
# ============================================================================
def create_dummy_dataset(
    num_samples: int = 1000,
    num_classes: int = NUM_CLASSES,
    img_size: Tuple[int, int] = IMG_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Creează un dataset dummy pentru testare rapidă.
    În producție, înlocuiește cu date reale.
    """
    print(f"\n📊 Creare dataset dummy pentru testare")
    print(f"   Samples: {num_samples}")
    print(f"   Classes: {num_classes}")
    print(f"   Image size: {img_size}")
    
    # Generăm date random
    train_size = int(num_samples * 0.8)
    test_size = num_samples - train_size
    
    # Date de antrenare
    train_images = np.random.rand(train_size, *img_size, 3).astype(np.float32)
    train_labels = np.random.randint(0, num_classes, train_size).astype(np.int32)
    
    # Date de test
    test_images = np.random.rand(test_size, *img_size, 3).astype(np.float32)
    test_labels = np.random.randint(0, num_classes, test_size).astype(np.int32)
    
    # Creăm tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    
    print(f"✅ Dataset creat!")
    
    return train_ds, test_ds

def load_real_dataset_from_url(
    dataset_url: str = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Încarcă un dataset real de imagini.
    Exemplu cu CIFAR-10 (disponibil în Keras).
    """
    print(f"\n📥 Încărcare dataset real: CIFAR-10")
    
    try:
        # Încărcăm CIFAR-10 (similar cu descărcarea de pe HuggingFace)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        # Redimensionăm imaginile la dimensiunea cerută de model
        x_train = tf.image.resize(x_train, IMG_SIZE).numpy()
        x_test = tf.image.resize(x_test, IMG_SIZE).numpy()
        
        # Normalizare
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        
        # Flatten labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        # Creăm tf.data.Dataset
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        
        print(f"✅ Dataset încărcat!")
        print(f"   Train samples: {len(x_train)}")
        print(f"   Test samples: {len(x_test)}")
        print(f"   Classes: 10 (CIFAR-10)")
        
        return train_ds, test_ds
        
    except Exception as e:
        print(f"⚠️ Nu s-a putut încărca CIFAR-10: {e}")
        print("   Se creează dataset dummy...")
        return create_dummy_dataset(num_classes=10)

def preprocess_loaded_data(
    train_ds: tf.data.Dataset, 
    test_ds: tf.data.Dataset,
    batch_size: int = 32,
    shuffle_buffer: int = 1000
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Preprocesează dataset-urile cu batching și shuffling.
    """
    print(f"\n🔄 Preprocesare date...")
    print(f"   Batch size: {batch_size}")
    
    # Preprocesare pentru antrenare
    train_ds = train_ds.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Preprocesare pentru test (fără shuffle)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print("✅ Preprocesare completă!")
    
    return train_ds, test_ds

# ============================================================================
# 3. FUNCȚIE PENTRU COMPILAREA MODELULUI
# ============================================================================
def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
    optimizer_name: str = 'adam'
) -> tf.keras.Model:
    """
    Compilează modelul cu setările optime.
    """
    print(f"\n⚙️ Compilare model...")
    print(f"   Optimizer: {optimizer_name}")
    print(f"   Learning rate: {learning_rate}")
    
    # Selectăm optimizer-ul
    if optimizer_name.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compilăm modelul
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    print("✅ Model compilat!")
    
    return model

# ============================================================================
# 4. FUNCȚIE PENTRU ANTRENAREA REȚELEI NEURONALE (DIN TEMPLATE)
# ============================================================================
def train_neural_network(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset = None,
    epochs: int = 5,
    callbacks: list = None,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Antrenează o rețea neuronală pe un dataset furnizat.
    Funcție generică compatibilă cu orice arhitectură Keras.
    """
    if not isinstance(model, tf.keras.Model):
        raise TypeError("Modelul trebuie să fie o instanță tf.keras.Model")
    
    # Verificare compilare
    try:
        _ = model.optimizer
        is_compiled = True
    except (AttributeError, ValueError):
        is_compiled = False
    
    if not is_compiled:
        raise ValueError(
            "Modelul trebuie să fie compilat înainte de antrenare. "
            "Folosiți compile_model(model)"
        )
    
    if callbacks is None:
        callbacks = []
    
    print(f"\n🚀 Începe antrenarea pentru {epochs} epoci...")
    
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )
    
    print(f"✅ Antrenare finalizată!")
    
    return {
        'history': history.history,
        'epochs_completed': len(history.history['loss'])
    }

# ============================================================================
# 5. FUNCȚII PENTRU EXTRAGEREA/SETAREA PONDERILOR (DIN TEMPLATE)
# ============================================================================
def get_model_weights(model: tf.keras.Model) -> list:
    """Extrage ponderile (weights) din model."""
    return model.get_weights()

def set_model_weights(model: tf.keras.Model, weights: list) -> None:
    """Setează ponderile (weights) în model."""
    model.set_weights(weights)

# ============================================================================
# 6. FUNCȚIE PENTRU CALCULAREA METRICILOR (DIN TEMPLATE)
# ============================================================================
def calculate_metrics(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculează metrici de evaluare: accuracy, precision, recall, f1_score.
    """
    print("\n📊 Calculare metrici...")
    
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
    
    print("✅ Metrici calculate!")
    
    return metrics

# ============================================================================
# 7. FUNCȚIE DE VALIDARE A MODELULUI (DIN TEMPLATE)
# ============================================================================
def validate_model_structure(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Validează și returnează informații despre structura modelului.
    """
    try:
        _ = model.optimizer
        is_compiled = True
    except (AttributeError, ValueError):
        is_compiled = False
    
    info = {
        'total_params': model.count_params(),
        'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.size(w).numpy() for w in model.non_trainable_weights]),
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
# 8. FUNCȚIE PENTRU FINE-TUNING (UNFREEZE LAYERS)
# ============================================================================
def unfreeze_model_layers(
    model: tf.keras.Model,
    num_layers_to_unfreeze: int = 20
) -> tf.keras.Model:
    """
    Deblochează ultimele layere ale modelului pentru fine-tuning.
    """
    print(f"\n🔓 Deblocare layere pentru fine-tuning...")
    print(f"   Layere de deblocheat: {num_layers_to_unfreeze}")
    
    # Înghețăm toate layerele mai întâi
    for layer in model.layers:
        layer.trainable = False
    
    # Deblochează ultimele layere
    for layer in model.layers[-num_layers_to_unfreeze:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    print(f"✅ Layere antrenabile: {trainable_count}/{len(model.layers)}")
    
    return model

# ============================================================================
# EXEMPLU DE UTILIZARE COMPLETĂ
# ============================================================================
if __name__ == "__main__":
    
    print("=" * 80)
    print("🤗 MODEL PRE-ANTRENAT (HUGGINGFACE-STYLE) + TEMPLATE INTEGRATION")
    print("=" * 80)
    
    try:
        # ====================================================================
        # PASUL 1: Descărcare model pre-antrenat
        # ====================================================================
        model = download_pretrained_model(
            model_name='MobileNetV2',  # Schimbă cu: ResNet50, VGG16, etc.
            num_classes=10,  # Pentru CIFAR-10
            weights='imagenet'
        )
        
        # ====================================================================
        # PASUL 2: Încărcare date
        # ====================================================================
        # Opțiune 1: Dataset real (CIFAR-10)
        train_ds, test_ds = load_real_dataset_from_url()
        
        # Opțiune 2: Dataset dummy (pentru testare rapidă)
        # train_ds, test_ds = create_dummy_dataset(num_classes=10)
        
        # ====================================================================
        # PASUL 3: Preprocesare
        # ====================================================================
        train_ds, test_ds = preprocess_loaded_data(
            train_ds, 
            test_ds, 
            batch_size=32,
            shuffle_buffer=1000
        )
        
        # ====================================================================
        # PASUL 4: Compilare model
        # ====================================================================
        model = compile_model(
            model, 
            learning_rate=1e-3,
            optimizer_name='adam'
        )
        
        # ====================================================================
        # PASUL 5: Validare structură (FUNCȚIE DIN TEMPLATE)
        # ====================================================================
        print("\n📋 --- Informații Model ---")
        model_info = validate_model_structure(model)
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # ====================================================================
        # PASUL 6: Antrenare inițială (FUNCȚIE DIN TEMPLATE)
        # ====================================================================
        history = train_neural_network(
            model=model,
            train_dataset=train_ds,
            validation_dataset=test_ds,
            epochs=2,  # Antrenare rapidă pentru top layers
            verbose=1
        )
        
        print(f"\n✅ Antrenare inițială finalizată după {history['epochs_completed']} epoci")
        
        # ====================================================================
        # PASUL 7: Extragere ponderi (FUNCȚIE DIN TEMPLATE)
        # ====================================================================
        weights = get_model_weights(model)
        print(f"\n💾 Ponderi extrase: {len(weights)} tensori")
        print(f"   Primul tensor shape: {weights[0].shape}")
        print(f"   Ultimul tensor shape: {weights[-1].shape}")
        
        # ====================================================================
        # PASUL 8: Calculare metrici (FUNCȚIE DIN TEMPLATE)
        # ====================================================================
        metrics = calculate_metrics(model, test_ds, average='macro')
        print("\n📊 --- Metrici după antrenare inițială ---")
        for metric_name, value in metrics.items():
            print(f"   {metric_name}: {value:.4f}")
        
        # ====================================================================
        # PASUL 9: Fine-tuning (opțional)
        # ====================================================================
        print("\n" + "=" * 80)
        print("🔥 FINE-TUNING (Deblocare layere)")
        print("=" * 80)
        
        model = unfreeze_model_layers(model, num_layers_to_unfreeze=30)
        
        # Re-compilăm cu learning rate mai mic pentru fine-tuning
        model = compile_model(model, learning_rate=1e-4, optimizer_name='adam')
        
        # Antrenare fine-tuning
        history_ft = train_neural_network(
            model=model,
            train_dataset=train_ds,
            validation_dataset=test_ds,
            epochs=2,
            verbose=1
        )
        
        # Metrici după fine-tuning
        metrics_ft = calculate_metrics(model, test_ds, average='macro')
        print("\n📊 --- Metrici după fine-tuning ---")
        for metric_name, value in metrics_ft.items():
            print(f"   {metric_name}: {value:.4f}")
        
        # ====================================================================
        # PASUL 10: Salvare model
        # ====================================================================
        model_save_path = "./saved_pretrained_model.h5"
        model.save(model_save_path)
        print(f"\n💾 Model salvat la: {model_save_path}")
        
        # ====================================================================
        # PASUL 11: Test de setare ponderi (FUNCȚIE DIN TEMPLATE)
        # ====================================================================
        print("\n🔄 Test setare ponderi...")
        new_weights = get_model_weights(model)
        set_model_weights(model, new_weights)
        print("✅ Ponderi setate cu succes!")
        
        print("\n" + "=" * 80)
        print("🎉 PROCES COMPLET FINALIZAT CU SUCCES!")
        print("=" * 80)
        print("\n📝 REZUMAT:")
        print(f"   - Model: MobileNetV2 fine-tuned")
        print(f"   - Dataset: CIFAR-10 (50,000 train, 10,000 test)")
        print(f"   - Accuracy inițială: {metrics['accuracy']:.4f}")
        print(f"   - Accuracy după fine-tuning: {metrics_ft['accuracy']:.4f}")
        print(f"   - Îmbunătățire: {(metrics_ft['accuracy'] - metrics['accuracy'])*100:.2f}%")
            
    except Exception as e:
        import traceback
        print(f"\n❌ Eșec în execuția completă. Eroare: {e}")
        print("\nTraceback complet:")
        traceback.print_exc()