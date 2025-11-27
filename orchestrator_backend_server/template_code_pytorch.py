"""
Template pentru antrenarea rețelelor neuronale în medii Federated Learning - PYTORCH VERSION
Compatibil cu orice arhitectură PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any, Callable
import json
from pathlib import Path
from PIL import Image
import os


# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================

class ImageFolderDataset(Dataset):
    """Custom Dataset for loading images from directory structure."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = []
        
        # Load all images and labels
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                continue
                
            self.classes.append(class_name)
            
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_file, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# 1. FUNCȚII PENTRU EXTRAGEREA DATELOR
# ============================================================================

def load_train_test_data() -> Tuple[Dataset, Dataset]:
    """
    Funcție care trebuie completată de utilizator pentru a încărca datele.
    
    Returns:
        Tuple[Dataset, Dataset]: (train_dataset, test_dataset)
    """
    """Implementare exemplu pentru MNIST"""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    return train_dataset, test_dataset


def preprocess_transform():
    """
    Returnează transformările de preprocesare pentru imagini.
    """
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x if x.shape[0] == 1 else x[0:1])  # Ensure single channel
    ])


def load_client_data(data_path: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Funcție pentru încărcarea datelor în FL simulator (AGNOSTIC).
    
    Această funcție este apelată de fd_simulator pentru fiecare client.
    Implementarea este COMPLETĂ - simulatorul NU aplică nicio preprocesare proprie!
    
    Args:
        data_path: Path către directorul cu date (ex: "clean_data" sau "clean_data_poisoned")
        batch_size: Dimensiunea batch-ului
        
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader) complet preprocesate
        
    Important:
        - Datele returnate TREBUIE să fie gata pentru antrenare (preprocessed, batched)
        - Simulatorul NU va aplica nicio transformare suplimentară
        - Această funcție oferă control TOTAL asupra preprocesării
    """
    data_path = Path(data_path)
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Data directories not found: {data_path}")
    
    # Define transforms
    transform = preprocess_transform()
    
    # Create datasets
    train_dataset = ImageFolderDataset(train_dir, transform=transform)
    test_dataset = ImageFolderDataset(test_dir, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, test_loader


def download_data(output_dir: str = "clean_data"):
    """
    Descarcă, preprocesează și salvează datele în două directoare separate:
    clean_data/train/<clasă>/ și clean_data/test/<clasă>/.
    Această funcție este apelată de orchestrator.
    """
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading and saving data to {output_dir}...")
    
    # 1️⃣ Încarcă datele
    train_dataset, test_dataset = load_train_test_data()
    
    # 2️⃣ Convertește în numpy arrays
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for img, label in train_dataset:
        X_train.append(img.numpy())
        y_train.append(label)
    
    for img, label in test_dataset:
        X_test.append(img.numpy())
        y_test.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    num_classes = len(np.unique(y_train))
    
    # 3️⃣ Salvează imaginile în directoare
    def save_images(X, y, base_dir):
        for i, (img_array, label) in enumerate(zip(X, y)):
            class_dir = base_dir / str(int(label))
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Convertește tensorul în imagine PIL
            # img_array shape is (C, H, W)
            img = (img_array * 255).astype(np.uint8)
            if img.shape[0] == 1:
                img = np.squeeze(img, axis=0)
            elif len(img.shape) == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            
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
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader = None,
    epochs: int = 10,
    device: str = None,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Antrenează o rețea neuronală pe un dataset furnizat.
    Funcție generică compatibilă cu orice arhitectură PyTorch.
    
    Args:
        model: Model PyTorch (nn.Module)
        train_loader: DataLoader pentru antrenare
        validation_loader: DataLoader pentru validare (opțional)
        epochs: Număr de epoci
        device: Device pentru antrenare ('cpu', 'cuda:1', sau None pentru auto-detect)
        verbose: Nivel de detaliere (0, 1, sau 2)
    
    Returns:
        Dict cu istoricul antrenării
    """
    if not isinstance(model, nn.Module):
        raise TypeError("Modelul trebuie să fie o instanță nn.Module")
    
    # Auto-detect device (cu fallback safe pentru multi-GPU)
    if device is None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                device = 'cuda:1'  # Folosește GPU 1 dacă disponibil
            else:
                device = 'cuda:0'
        else:
            device = 'cpu'
    
    # Creează obiect torch.device pentru comparații precise
    device_obj = torch.device(device)
    
    model = model.to(device_obj)
    model.train()
    
    # Get optimizer and criterion from model (should be set by _model_compile)
    if not hasattr(model, 'optimizer'):
        raise ValueError("Model must have 'optimizer' attribute. Call _model_compile() first.")
    if not hasattr(model, 'criterion'):
        raise ValueError("Model must have 'criterion' attribute. Call _model_compile() first.")
    
    optimizer = model.optimizer
    criterion = model.criterion

    # Verifică și mută parametrii modelului (FIXAT: comparație full device)
    for param in model.parameters():
        if param.device != device_obj:
            param.data = param.data.to(device_obj)
            if param.device != device_obj:  # Re-check după mutare
                raise RuntimeError(f"Parameter on wrong device: {param.device}, expected: {device_obj}")
    
    # Verifică și mută starea optimizatorului (FIXAT: full device + dtype match cu params)
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, {})
            for k, v in list(state.items()):  # list() pentru a evita modificări în iterare
                if isinstance(v, torch.Tensor):
                    target_dtype = p.dtype  # Starea trebuie să match-uiască dtype-ul paramului
                    if v.device != device_obj or v.dtype != target_dtype:
                        state[k] = v.to(device=device_obj, dtype=target_dtype)
                        if state[k].device != device_obj or state[k].dtype != target_dtype:
                            raise RuntimeError(f"Optimizer state {k} on wrong device/dtype: {state[k].device}/{state[k].dtype}, expected: {device_obj}/{target_dtype}")
    
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device_obj), targets.to(device_obj)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        
        # Validation phase
        if validation_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in validation_loader:
                    inputs, targets = inputs.to(device_obj), targets.to(device_obj)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_epoch_loss = val_loss / len(validation_loader)
            val_epoch_acc = 100. * val_correct / val_total
            
            history['val_loss'].append(val_epoch_loss)
            history['val_accuracy'].append(val_epoch_acc)
            
            if verbose >= 1:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"loss: {epoch_loss:.4f} - acc: {epoch_acc:.2f}% - "
                      f"val_loss: {val_epoch_loss:.4f} - val_acc: {val_epoch_acc:.2f}%")
        else:
            if verbose >= 1:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.2f}%")
    
    return history


# ============================================================================
# 3. FUNCȚII PENTRU MANIPULAREA PONDERILOR
# ============================================================================

def get_model_weights(model: nn.Module) -> list:
    """
    Extrage ponderile modelului sub formă de liste NumPy.
    
    Args:
        model: Model PyTorch
        
    Returns:
        List de array-uri NumPy reprezentând ponderile
    """
    return [param.data.cpu().numpy() for param in model.parameters()]


def set_model_weights(model: nn.Module, weights: list) -> None:
    """
    Setează ponderile modelului din liste NumPy.
    
    Args:
        model: Model PyTorch
        weights: Listă de array-uri NumPy
    """
    with torch.no_grad():
        for param, weight in zip(model.parameters(), weights):
            param.data = torch.from_numpy(weight).to(param.device)


# ============================================================================
# 4. FUNCȚIE PENTRU CALCULAREA METRICILOR
# ============================================================================

def calculate_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    average: str = 'weighted',
    device: str = None
) -> Dict[str, float]:
    """
    Calculează metrici de clasificare pe un dataset de test.
    
    Args:
        model: Model PyTorch antrenat
        test_loader: DataLoader pentru date de test
        average: Tipul de medie pentru metrici ('weighted', 'macro', 'micro')
        device: Device pentru evaluare
        
    Returns:
        Dict cu metrici (accuracy, precision, recall, f1_score)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    
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
    model: nn.Module,
    filepath: str,
    save_weights: bool = True
) -> None:
    """
    Salvează configurația completă a modelului în format .pth.
    
    Args:
        model: Model PyTorch
        filepath: Calea către fișierul de salvare (ex: 'model.pth')
        save_weights: Dacă True, salvează și ponderile
    """
    if not filepath.endswith('.pth') and not filepath.endswith('.pt'):
        filepath += '.pth'
    
    if save_weights:
        # Salvează state dict complet (include și optimizer dacă există)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
        }
        
        if hasattr(model, 'optimizer'):
            checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
    else:
        # Salvează doar arhitectura (nu e standard în PyTorch, salvăm doar structura)
        config = {
            'model_class': model.__class__.__name__,
            'model_config': str(model)
        }
        
        with open(filepath.replace('.pth', '_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"Model salvat în: {filepath}")


def load_model_config(filepath: str) -> nn.Module:
    """
    Încarcă configurația modelului din fișier .pth.
    
    Args:
        filepath: Calea către fișierul salvat
        
    Returns:
        Model PyTorch încărcat
    """
    if not filepath.endswith('.pth') and not filepath.endswith('.pt'):
        filepath += '.pth'
    
    try:
        # Detectăm dispozitivul
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Încărcăm checkpoint-ul
        checkpoint = torch.load(filepath, map_location=device)
        
        # Creăm modelul
        model = SimpleMNISTModel()
        
        # Mutăm modelul pe dispozitiv
        model = model.to(device)
        
        # Încărcăm state_dict-ul modelului
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Compilăm modelul pentru a seta optimizer și criterion
        model = _model_compile(model)
        
        # Dacă există stare pentru optimizator, o încărcăm
        if 'optimizer_state_dict' in checkpoint:
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Mutăm starea optimizatorului pe dispozitiv
            for state in model.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        
        print(f"Model încărcat din: {filepath} pe dispozitivul {device}")
        return model
    except Exception as e:
        print(f"Eroare la încărcarea modelului: {e}")
        raise

# ============================================================================
# 6. FUNCȚIE AUXILIARĂ PENTRU SALVAREA PONDERILOR SEPARATE
# ============================================================================

def save_weights_only(model: nn.Module, filepath: str) -> None:
    """
    Salvează doar ponderile modelului (fără arhitectură).
    Util pentru transferul rapid de ponderi în Federated Learning.
    
    Args:
        model: Model PyTorch
        filepath: Calea către fișier (ex: 'weights.pth')
    """
    torch.save(model.state_dict(), filepath)
    print(f"Ponderi salvate în: {filepath}")


def load_weights_only(model: nn.Module, filepath: str) -> None:
    """
    Încarcă doar ponderile în model (arhitectura trebuie să existe deja).
    
    Args:
        model: Model PyTorch cu arhitectura corectă
        filepath: Calea către fișierul cu ponderi
    """
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    print(f"Ponderi încărcate din: {filepath}")


# ============================================================================
# 7. FUNCȚIE DE VALIDARE A MODELULUI
# ============================================================================

def validate_model_structure(model: nn.Module) -> Dict[str, Any]:
    """
    Validează și returnează informații despre structura modelului.
    Util pentru verificarea compatibilității înainte de antrenare.
    
    Args:
        model: Model PyTorch
        
    Returns:
        Dict cu informații despre model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layers_count': len(list(model.modules())),
        'is_compiled': hasattr(model, 'optimizer') and hasattr(model, 'criterion')
    }
    
    if hasattr(model, 'optimizer'):
        info['optimizer'] = model.optimizer.__class__.__name__
    if hasattr(model, 'criterion'):
        info['loss'] = model.criterion.__class__.__name__
    
    return info


# ============================================================================
# EXEMPLU DE UTILIZARE
# ============================================================================

def _model_compile(model: nn.Module) -> nn.Module:
    """
    Configurează modelul cu optimizer și loss function.
    
    Args:
        model: Model PyTorch necompilat
        
    Returns:
        Model PyTorch cu optimizer și criterion setate
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.criterion = nn.CrossEntropyLoss().to(device)
    return model


def get_loss_type() -> str:
    """
    Returnează tipul funcției de loss folosită.
    
    Returns:
        str: Tipul funcției de loss
    """
    return 'CrossEntropyLoss'


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


# ============================================================================
# MODEL DEFINITION EXAMPLE
# ============================================================================

class SimpleMNISTModel(nn.Module):
    """Exemplu de model simplu pentru MNIST"""
    
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    """
    Exemplu complet de utilizare a template-ului PyTorch.
    """
    
    print("="*70)
    print("PyTorch Template Test")
    print("="*70)
    
    # Pasul 1: Creare model
    print("\n1. Creating model...")
    model = SimpleMNISTModel()
    model = _model_compile(model)
    
    # Pasul 2: Validare structură
    print("\n2. Validating model structure...")
    model_info = validate_model_structure(model)
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Pasul 3: Încărcare date
    print("\n3. Loading data...")
    train_dataset, test_dataset = load_train_test_data()
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # Pasul 4: Antrenare (few epochs for testing)
    print("\n4. Training model...")
    history = train_neural_network(
        model=model,
        train_loader=train_loader,
        validation_loader=test_loader,
        epochs=2,
        verbose=1
    )
    
    # Pasul 5: Extragere ponderi
    print("\n5. Extracting weights...")
    weights = get_model_weights(model)
    print(f"   Număr de layere cu ponderi: {len(weights)}")
    
    # Pasul 6: Calculare metrici
    print("\n6. Calculating metrics...")
    metrics = calculate_metrics(model, test_loader)
    for metric_name, value in metrics.items():
        print(f"   {metric_name}: {value:.4f}")
    
    # Pasul 7: Salvare model
    print("\n7. Saving model...")
    filepath = "simple_mnist_pytorch.pth"
    save_model_config(model, filepath)
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)