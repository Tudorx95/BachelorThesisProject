# ============================================================
# 🧠 ANTRENARE ResNet18 CIFAR-10 + UPLOAD HUGGINGFACE
# ============================================================
# Script pentru Google Colab - compatibil cu template_antrenare_pytorch.py
# Activeaza GPU: Runtime > Change runtime type > GPU
# ============================================================

# ⚙️ CONFIGUREAZA AICI ⚙️
HF_TOKEN = ""  # Token HuggingFace (de la https://huggingface.co/settings/tokens)
HF_USERNAME = ""  # Username-ul tau HuggingFace
MODEL_NAME = "resnet18-cifar10"  # Numele modelului
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
PRIVATE_REPO = False
# ============================================================

# Instalare
!pip install -q huggingface_hub

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
from pathlib import Path
from huggingface_hub import login, create_repo, upload_folder
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATIE (compatibila cu template-ul tau)
# ============================================================
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
NUM_CLASSES = 10
IMG_SIZE = (32, 32)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Login HuggingFace
if not HF_TOKEN:
    from getpass import getpass
    HF_TOKEN = getpass("🔑 Introdu HuggingFace Token: ")
if not HF_USERNAME:
    HF_USERNAME = input("👤 Introdu HuggingFace Username: ")

login(token=HF_TOKEN)
REPO_NAME = f"{HF_USERNAME}/{MODEL_NAME}"

print(f"\n{'='*60}")
print(f"🖥️  Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
print(f"📁 Repo: {REPO_NAME}")
print(f"📊 Dataset: CIFAR-10 | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"{'='*60}\n")

# ============================================================
# CREARE MODEL ResNet18 PENTRU CIFAR-10
# ============================================================
def create_resnet18_cifar10():
    """
    Creeaza ResNet18 adaptat pentru CIFAR-10 (32x32 imagini).
    Modificari fata de ResNet18 standard:
    - conv1: kernel 3x3 in loc de 7x7
    - fara maxpool (imaginile sunt prea mici)
    - fc layer cu 10 clase
    """
    model = torchvision.models.resnet18(weights=None)
    
    # Modifică primul layer pentru CIFAR-10 (32x32 în loc de 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Elimină maxpool pentru imagini mici
    
    # Modifică ultimul layer pentru 10 clase
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model

model = create_resnet18_cifar10().to(device)

# Numara parametrii
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model ResNet18 creat")
print(f"   Total parametri: {total_params:,}")
print(f"   Trainable: {trainable_params:,}\n")

# ============================================================
# INCARCARE DATE CIFAR-10
# ============================================================
# Transformari cu augmentare pentru antrenare
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"✅ Date CIFAR-10 incarcate")
print(f"   Train: {len(train_dataset):,} imagini")
print(f"   Test: {len(test_dataset):,} imagini")
print(f"   Clase: {CIFAR10_CLASSES}\n")

# ============================================================
# ANTRENARE
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

print("🚀 Incepe antrenarea...\n" + "="*60)

for epoch in range(1, EPOCHS + 1):
    # --- TRAIN ---
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        correct += (output.argmax(1) == target).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    train_loss /= len(train_loader)
    train_acc = 100 * correct / total
    
    # --- EVAL ---
    model.eval()
    test_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    
    # Scheduler step
    scheduler.step()
    
    # Salveaza istoric
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    
    print(f"\nEpoca {epoch}/{EPOCHS}")
    print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    print("-"*60)

print(f"\n✅ Antrenare completa! Test Accuracy: {test_acc:.2f}%\n")

# ============================================================
# METRICI DETALIATE
# ============================================================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        output = model(data)
        y_pred.extend(output.argmax(1).cpu().numpy())
        y_true.extend(target.numpy())

y_true, y_pred = np.array(y_true), np.array(y_pred)

metrics = {
    'accuracy': float(accuracy_score(y_true, y_pred)),
    'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
    'recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
    'f1_score': float(f1_score(y_true, y_pred, average='macro', zero_division=0))
}

print("📊 Metrici finale:")
for k, v in metrics.items():
    print(f"   {k}: {v:.4f}")

# ============================================================
# GRAFICE
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history['train_loss'], 'o-', label='Train')
ax[0].plot(history['test_loss'], 'o-', label='Test')
ax[0].set_title('Loss'); ax[0].set_xlabel('Epoca'); ax[0].legend(); ax[0].grid(True)
ax[1].plot(history['train_acc'], 'o-', label='Train')
ax[1].plot(history['test_acc'], 'o-', label='Test')
ax[1].set_title('Accuracy (%)'); ax[1].set_xlabel('Epoca'); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()

# ============================================================
# SALVARE MODEL (format compatibil cu template-ul tau)
# ============================================================
save_dir = Path('./model_output')
save_dir.mkdir(exist_ok=True)

# Salvare model COMPLET intr-un singur fisier .pth
# Contine: state_dict + arhitectura + normalizare + metrici + config antrenare
MODEL_FILENAME = "ResNet18_CIFAR10.pth"

torch.save({
    'model_state_dict': model.state_dict(),
    'architecture': {
        'base': 'resnet18',
        'num_classes': NUM_CLASSES,
        'img_size': list(IMG_SIZE),
        'modifications': {
            'conv1': {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            'maxpool': 'Identity',
            'fc': {'in_features': 512, 'out_features': NUM_CLASSES}
        }
    },
    'normalization': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010]
    },
    'classes': CIFAR10_CLASSES,
    'metrics': metrics,
    'training_config': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'scheduler_params': {'step_size': 5, 'gamma': 0.5}
    },
    'history': history
}, save_dir / MODEL_FILENAME)

print(f"\n✅ Model complet salvat: {MODEL_FILENAME}")
print(f"   Contine: state_dict + arhitectura + normalizare + metrici + config")

# Config JSON (pentru afisare pe HuggingFace)
config = {
    'model_name': 'ResNet18',
    'dataset': 'CIFAR-10',
    'num_classes': NUM_CLASSES,
    'img_size': list(IMG_SIZE),
    'classes': CIFAR10_CLASSES,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'optimizer': 'Adam',
    'scheduler': 'StepLR',
    'architecture_changes': [
        'conv1: 3x3 kernel instead of 7x7',
        'maxpool: removed (Identity)',
        'fc: 512 -> 10 classes'
    ]
}
with open(save_dir / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Metrici JSON (pentru afisare pe HuggingFace)
with open(save_dir / 'metrics.json', 'w') as f:
    json.dump({'final_metrics': metrics, 'history': history}, f, indent=2)

# README pentru HuggingFace
readme = f"""---
library_name: pytorch
tags:
- image-classification
- pytorch
- resnet
- cifar10
datasets:
- cifar10
metrics:
- accuracy
---

# ResNet18 pentru CIFAR-10

Model ResNet18 adaptat si antrenat pe CIFAR-10.

## Performanta

| Metrica | Valoare |
|---------|---------|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1-Score | {metrics['f1_score']:.4f} |

## Utilizare cu template_antrenare_pytorch.py

```python
# In template, seteaza:
HUGGINGFACE_REPO_ID = "{REPO_NAME}"
MODEL_FILENAME = "{MODEL_FILENAME}"

# Modelul se incarca automat in create_model()
```

## Utilizare directa

```python
import torch
import torchvision
import torch.nn as nn

# Incarca checkpoint-ul
checkpoint = torch.load('{MODEL_FILENAME}', map_location='cpu')

# Reconstruieste arhitectura din config
arch = checkpoint['architecture']
model = torchvision.models.resnet18(weights=None)
mod = arch['modifications']
model.conv1 = nn.Conv2d(3, 64, **mod['conv1'])
model.maxpool = nn.Identity()
model.fc = nn.Linear(mod['fc']['in_features'], mod['fc']['out_features'])

# Incarca ponderile
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Clase CIFAR-10

{', '.join([f'`{i}: {c}`' for i, c in enumerate(CIFAR10_CLASSES)])}

## Antrenare

- **Epochs:** {EPOCHS}
- **Batch Size:** {BATCH_SIZE}
- **Learning Rate:** {LEARNING_RATE}
- **Optimizer:** Adam
- **Scheduler:** StepLR (step=5, gamma=0.5)
- **Augmentare:** RandomCrop, RandomHorizontalFlip
"""

with open(save_dir / 'README.md', 'w') as f:
    f.write(readme)

print(f"✅ Toate fisierele salvate in {save_dir}/")

# ============================================================
# UPLOAD PE HUGGINGFACE
# ============================================================
print("\n📤 Upload pe HuggingFace...")
create_repo(REPO_NAME, repo_type="model", private=PRIVATE_REPO, exist_ok=True)
upload_folder(folder_path=str(save_dir), repo_id=REPO_NAME, repo_type="model")

print(f"\n{'='*60}")
print(f"✅ SUCCES! Model uploadat pe HuggingFace")
print(f"{'='*60}")
print(f"\n🔗 Link: https://huggingface.co/{REPO_NAME}")
print(f"\n📝 Pentru a folosi in template_antrenare_pytorch.py:")
print(f'   HUGGINGFACE_REPO_ID = "{REPO_NAME}"')
print(f'   MODEL_FILENAME = "{MODEL_FILENAME}"')
print(f"{'='*60}")