# 🎯 FD Simulator Agnostic - Documentație Completă

## 📋 Overview

**fd_simulator_agnostic.py** este o versiune **complet agnostică** a simulatorului FL care:
- ❌ **NU aplică** nicio preprocesare hardcoded
- ❌ **NU impune** label_mode, color_mode sau alte setări
- ✅ **DELEGE** toată logica de date către template
- ✅ **OFERĂ** control total utilizatorului

---

## 🔄 Ce S-a Schimbat?

### Înainte (fd_simulatorV2.py) - ❌ OPINIONATED

```python
# Simulator IMPUNEA preprocesări hardcoded:

def _load_tensorflow_data(train_dir, test_dir, batch_size):
    # ❌ Hardcoded: label_mode='categorical'
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode='categorical',  # ← IMPUS de simulator!
        color_mode='rgb',           # ← IMPUS de simulator!
        ...
    )
    
    # ❌ Hardcoded: normalizare default
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label
    train_ds = train_ds.map(normalize)  # ← IMPUSĂ de simulator!
```

**Probleme**:
- Double one-hot encoding (simulator + template)
- Nu poți folosi sparse labels
- Nu poți customiza preprocesarea
- Nu poți folosi alte transformări

### După (fd_simulator_agnostic.py) - ✅ AGNOSTIC

```python
# Simulator DELEGE totul către template:

def load_data(data_path, framework, batch_size):
    # ✅ Priority 1: Template's complete custom loading
    if TEMPLATE_FUNCS.has_function('load_client_data'):
        return TEMPLATE_FUNCS.get_function('load_client_data')(
            str(data_path), batch_size
        )
    
    # ✅ Priority 2: Template's general loading
    elif TEMPLATE_FUNCS.has_function('load_train_test_data'):
        train_ds, test_ds = TEMPLATE_FUNCS.get_function('load_train_test_data')()
        
        if TEMPLATE_FUNCS.has_function('preprocess_loaded_data'):
            train_ds, test_ds = TEMPLATE_FUNCS.get_function('preprocess_loaded_data')(
                train_ds, test_ds
            )
        return train_ds, test_ds
    
    # ✅ Error: template TREBUIE să implementeze data loading
    else:
        raise SimulationError("Template must implement data loading!")
```

**Beneficii**:
- Zero preprocesări hardcoded
- Control TOTAL asupra datelor
- Flexibilitate completă
- Nu mai există conflicte label_mode

---

## 🔧 Cum Funcționează?

### Fluxul Complet Agnostic

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. FD Simulator - AGNOSTIC Layer                                │
│    fd_simulator_agnostic.py                                      │
│                                                                  │
│    NO hardcoded preprocessing!                                  │
│    NO label_mode decisions!                                     │
│    NO color_mode defaults!                                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
             Calls: load_client_data(data_path, batch_size)
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Template - USER CONTROL Layer                                │
│    template_code_agnostic.py                                     │
│                                                                  │
│    def load_client_data(data_path, batch_size):                 │
│        # USER decides:                                           │
│        # - label_mode='int' or 'categorical'                     │
│        # - color_mode='grayscale' or 'rgb'                       │
│        # - image_size=(28, 28) or (224, 224)                     │
│        # - normalization: /255.0 or standard scaler              │
│        # - augmentation: yes or no                               │
│        # - one-hot encoding: manual or automatic                 │
│        #                                                          │
│        # EVERYTHING is controlled by USER!                       │
│                                                                  │
│        train_ds = image_dataset_from_directory(...)              │
│        train_ds = train_ds.map(custom_preprocess)                │
│        return train_ds, test_ds                                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                  Returns: (train_ds, test_ds)
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. FD Simulator - Uses Data As-Is                               │
│                                                                  │
│    model.fit(train_ds)  ← NO modifications!                     │
│    model.evaluate(test_ds)  ← Data used directly!               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 Template Requirements

### Funcție OBLIGATORIE: `load_client_data`

```python
def load_client_data(data_path: str, batch_size: int = 32) -> Tuple:
    """
    Încarcă și preprocesează date pentru FL clients.
    
    IMPORTANT:
    - Această funcție oferă CONTROL COMPLET
    - Simulatorul NU va aplica nicio transformare suplimentară
    - Datele returnate TREBUIE să fie gata pentru antrenare
    
    Args:
        data_path: Path către date (ex: "clean_data")
        batch_size: Dimensiunea batch-ului
        
    Returns:
        (train_ds, test_ds): Datasets complet preprocesate
    """
    # YOUR IMPLEMENTATION HERE
    pass
```

### Exemplu Complet (TensorFlow)

```python
def load_client_data(data_path: str, batch_size: int = 32):
    """Load data for FL simulation with FULL control."""
    from pathlib import Path
    import tensorflow as tf
    
    data_path = Path(data_path)
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    # Step 1: Load from directories
    # YOU decide label_mode, color_mode, image_size!
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(28, 28),        # ✅ YOUR choice
        batch_size=batch_size,
        color_mode='grayscale',     # ✅ YOUR choice
        label_mode='int',           # ✅ YOUR choice - no double one-hot!
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
    
    # Step 2: Apply YOUR preprocessing
    def my_preprocess(image, label):
        # ✅ YOUR normalization
        image = tf.cast(image, tf.float32) / 255.0
        
        # ✅ YOUR encoding strategy
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, 10)  # ONE-HOT only once!
        
        return image, label
    
    train_ds = train_ds.map(my_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(my_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Step 3: Optimization (optional)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds
```

### Exemplu Complet (PyTorch)

```python
def load_client_data(data_path: str, batch_size: int = 32):
    """Load data for FL simulation with PyTorch."""
    from pathlib import Path
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    data_path = Path(data_path)
    
    # Step 1: Define YOUR transforms
    transform = transforms.Compose([
        transforms.Resize((28, 28)),        # ✅ YOUR size
        transforms.Grayscale(),             # ✅ YOUR color mode
        transforms.ToTensor(),              # ✅ YOUR conversion
        transforms.Normalize((0.5,), (0.5,))  # ✅ YOUR normalization
    ])
    
    # Step 2: Load datasets
    train_dataset = datasets.ImageFolder(
        data_path / "train",
        transform=transform
    )
    
    test_dataset = datasets.ImageFolder(
        data_path / "test",
        transform=transform
    )
    
    # Step 3: Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader
```

---

## 🎯 Cazuri de Utilizare

### Caz 1: MNIST cu One-Hot Encoding

```python
def load_client_data(data_path: str, batch_size: int = 32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        Path(data_path) / "train",
        image_size=(28, 28),
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='int'  # ← Indici întregi
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        Path(data_path) / "test",
        image_size=(28, 28),
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='int'
    )
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, 10)  # One-hot manual
        return image, label
    
    train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds
```

**Model compile**:
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # ← One-hot labels
    metrics=['accuracy']
)
```

### Caz 2: CIFAR-10 cu Sparse Labels

```python
def load_client_data(data_path: str, batch_size: int = 32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        Path(data_path) / "train",
        image_size=(32, 32),
        batch_size=batch_size,
        color_mode='rgb',
        label_mode='int'  # ← Keep as integers
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        Path(data_path) / "test",
        image_size=(32, 32),
        batch_size=batch_size,
        color_mode='rgb',
        label_mode='int'
    )
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        # NO one-hot encoding - keep as integers!
        return image, label
    
    train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds
```

**Model compile**:
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # ← Integer labels
    metrics=['accuracy']
)
```

### Caz 3: ImageNet cu Data Augmentation

```python
def load_client_data(data_path: str, batch_size: int = 32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        Path(data_path) / "train",
        image_size=(224, 224),
        batch_size=batch_size,
        color_mode='rgb',
        label_mode='categorical'  # ← Automat one-hot pentru multe clase
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        Path(data_path) / "test",
        image_size=(224, 224),
        batch_size=batch_size,
        color_mode='rgb',
        label_mode='categorical'
    )
    
    # Data augmentation pentru training
    def augment_and_preprocess(image, label):
        # Augmentation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Normalization (ImageNet stats)
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        return image, label
    
    def preprocess_only(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return image, label
    
    train_ds = train_ds.map(augment_and_preprocess).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_only).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds
```

---

## ✅ Avantaje Simulatorul Agnostic

| Aspect | Simulator Opinionated (vechi) | Simulator Agnostic (nou) |
|--------|-------------------------------|---------------------------|
| **Preprocesare** | Hardcoded în simulator | Definită în template |
| **label_mode** | Impus de simulator | Controlat de user |
| **color_mode** | Impus de simulator | Controlat de user |
| **Normalization** | Default /255.0 | Custom în template |
| **Data Augmentation** | Impossible | Posibil în template |
| **One-Hot Encoding** | Risc de double encoding | Control complet |
| **Flexibilitate** | Low | HIGH |
| **Debugging** | Dificil (logic în 2 locuri) | Ușor (toată logica în template) |

---

## 🚨 Migrare de la Vechiul Simulator

### Ce Trebuie să Faci

1. **Adaugă funcția `load_client_data` în template**
   ```python
   def load_client_data(data_path: str, batch_size: int = 32):
       # Implementare completă aici
       pass
   ```

2. **Mutătoată preprocesarea din mental model la template**
   - Label mode (int vs categorical)
   - Color mode (grayscale vs rgb)
   - Normalization strategy
   - Data augmentation

3. **Verifică compatibilitatea cu loss function**
   ```python
   # Dacă folosești categorical_crossentropy:
   label_mode='int' + manual one-hot în preprocess
   
   # Dacă folosești sparse_categorical_crossentropy:
   label_mode='int' + NO one-hot
   ```

4. **Testează local înainte de deploy**

---

## 📋 Checklist Template Agnostic

- [ ] `load_client_data(data_path, batch_size)` implementată
- [ ] `label_mode` setat corect (int / categorical)
- [ ] `color_mode` setat corect (grayscale / rgb)
- [ ] Preprocesare definită explicit (normalization)
- [ ] One-hot encoding aplicat O SINGURĂ DATĂ (dacă e necesar)
- [ ] Loss function compatibil cu label format
- [ ] Testat local: `train_ds, test_ds = load_client_data("clean_data", 32)`
- [ ] Verificat shapes: `for imgs, lbls in train_ds.take(1): print(imgs.shape, lbls.shape)`

---

## 🎓 Best Practices

### 1. Definește helper functions în template

```python
def create_preprocess_function(num_classes: int):
    """Factory for preprocessing functions."""
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, num_classes)
        return image, label
    return preprocess

def load_client_data(data_path: str, batch_size: int = 32):
    train_ds = ...
    test_ds = ...
    
    preprocess_fn = create_preprocess_function(num_classes=10)
    train_ds = train_ds.map(preprocess_fn)
    test_ds = test_ds.map(preprocess_fn)
    
    return train_ds, test_ds
```

### 2. Documentează detaliile

```python
def load_client_data(data_path: str, batch_size: int = 32):
    """
    Load MNIST data for FL simulation.
    
    Data format:
    - Images: grayscale 28x28, normalized to [0, 1]
    - Labels: one-hot encoded (10 classes)
    - Batch size: configurable
    
    Preprocessing pipeline:
    1. Load from directory (label_mode='int')
    2. Normalize: /255.0
    3. One-hot encode labels manually
    4. Batch and prefetch
    """
    # Implementation
    pass
```

### 3. Validează output-ul

```python
def load_client_data(data_path: str, batch_size: int = 32):
    train_ds, test_ds = _internal_load(data_path, batch_size)
    
    # Validate shapes
    for images, labels in train_ds.take(1):
        assert images.shape == (batch_size, 28, 28, 1), "Wrong image shape"
        assert labels.shape == (batch_size, 10), "Wrong label shape"
        assert tf.reduce_min(images) >= 0.0, "Images not normalized"
        assert tf.reduce_max(images) <= 1.0, "Images not normalized"
    
    return train_ds, test_ds
```

---

**Versiune**: 3.0 - Fully Agnostic  
**Data**: October 25, 2025  
**Status**: ✅ PRODUCTION READY  
**Breaking Changes**: YES - requires template update
