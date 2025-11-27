# Ghid de Migrare: fd_simulator v1 → v2

## 📋 Rezumat Schimbări

### Schimbări Majore

| Aspect | v1 | v2 |
|--------|----|----|
| **Framework** | Doar TensorFlow | TensorFlow + PyTorch |
| **Parametri** | 9 parametri (paths complete) | 8 parametri (dir + nume relative) |
| **Logging** | Console only | Console + File în user dir |
| **Error Handling** | Basic exceptions | Graceful shutdown cu ErrorHandler |
| **Results Location** | Specificat explicit | Automat în task_dir/results/ |
| **Template Detection** | Manual | Automat (framework + funcții) |

---

## 🔄 Comparație Parametri

### v1 - Parametri Vechi
```bash
python fd_simulator.py \
    <test_file>              # Path complet către JSON results
    <N>                      # Număr clienți
    <M>                      # Număr clienți malițioși
    <NN_NAME_PATH>           # Path complet către model
    <data_folder>            # Path complet către date clean
    <alternative_data>       # Path complet către date poisoned
    <R>                      # Runde cu date poisoned
    <ROUNDS>                 # Total runde
    [--strategy STRATEGY]    # Strategie
    [--data_poisoning]       # Flag poisoning
    [--template PATH]        # Path template
```

### v2 - Parametri Noi
```bash
python fd_simulator_v2.py \
    <task_dir>               # Directorul task-ului
    <model_name>             # Numele fișierului model
    <clean_data_folder>      # Nume folder clean (relativ)
    <poisoned_data_folder>   # Nume folder poisoned (relativ)
    <N>                      # Număr clienți
    <M>                      # Număr clienți malițioși
    <R>                      # Runde cu date poisoned
    <ROUNDS>                 # Total runde
    [--strategy STRATEGY]    # Strategie
    [--data_poisoning]       # Flag poisoning
    [--template PATH]        # Path template
    [--results_file NAME]    # Nume fișier rezultate (opțional)
```

---

## 🔧 Exemple de Migrare

### Exemplu 1: Apel Simplu

**v1:**
```bash
python fd_simulator.py \
    /home/user/task_123/results/fl_clean.json \
    10 2 \
    /home/user/task_123/model.keras \
    /home/user/task_123/clean_data \
    /home/user/task_123/clean_data \
    2 5 \
    --strategy first \
    --template /home/user/task_123/template_code.py
```

**v2:**
```bash
python fd_simulator_v2.py \
    /home/user/task_123 \
    model.keras \
    clean_data \
    clean_data \
    10 2 2 5 \
    --strategy first \
    --template /home/user/task_123/template_code.py \
    --results_file fl_clean.json
```

**Diferențe:**
- ❌ Nu mai specificăm path-ul complet la results - se creează automat în `task_dir/results/`
- ❌ Nu mai repetăm `task_dir` de 5 ori
- ✅ Un singur parametru pentru directorul principal
- ✅ Nume relative pentru model și date

### Exemplu 2: În Orchestrator

**v1 - orchestrator.py:**
```python
# Step 8: Clean simulation
clean_results = user_dir / "results" / "fl_clean.json"
cmd = (
    f"{conda_activate} && "
    f"python {fl_script} {clean_results} {config['N']} {config['M']} "
    f"{model_path} {user_dir / 'clean_data'} {user_dir / 'clean_data'} "
    f"{config['R']} {config['ROUNDS']} --strategy {config['strategy']} --template {template_path}"
)
```

**v2 - orchestrator_fixed.py:**
```python
# Step 8: Clean simulation
cmd = (
    f"{conda_activate} && "
    f"python {fl_script} {user_dir} {config['NN_NAME']}.keras clean_data clean_data "
    f"{config['N']} {config['M']} {config['R']} {config['ROUNDS']} "
    f"--strategy {config['strategy']} --template {template_path} --results_file fl_clean.json"
)
clean_results = user_dir / "results" / "fl_clean.json"  # Path pentru parsing rezultate
```

**Beneficii:**
- ✅ Mai simplu de construit command-ul
- ✅ Mai puține erori la concatenarea path-urilor
- ✅ Rezultatele automat în locația corectă

---

## 📊 Schimbări în Output

### Structura Directoarelor

**v1:**
```
task_dir/
├── model.keras
├── clean_data/
├── template_code.py
└── results/
    └── fl_clean.json    # Trebuie specificat explicit
```

**v2:**
```
task_dir/
├── model.keras
├── clean_data/
├── template_code.py
└── results/
    ├── simulation.log      # NOU: Log complet
    ├── fl_clean.json       # Generat automat
    └── fl_poisoned.json    # Generat automat
```

### Format JSON Rezultate

**v1 - Minimal:**
```json
{
  "final_accuracy": 0.8789,
  "round_metrics_history": [...],
  "malicious_clients": [0, 1]
}
```

**v2 - Complet:**
```json
{
  "simulation_info": {
    "num_clients": 10,
    "num_malicious": 2,
    "rounds": 5,
    "strategy": "first",
    "data_poisoning": false,
    "framework": "tensorflow",         // NOU
    "timestamp": "2025-01-15T10:30:45"
  },
  "malicious_clients": [0, 1],
  "round_metrics_history": {...},
  "convergence_metrics": [...],        // NOU
  "weight_divergence": [...],          // NOU
  "round_times": [...],                // NOU
  "final_accuracy": 0.8789
}
```

---

## 🐍 Schimbări în Template

### Template TensorFlow

**Funcții identice între v1 și v2:**
- ✅ `download_data(output_dir)`
- ✅ `load_model_config(filepath)`
- ✅ `_model_compile(model)`
- ✅ `train_neural_network(model, train_data, test_data, epochs, verbose)`
- ✅ `get_model_weights(model)`
- ✅ `set_model_weights(model, weights)`
- ✅ `get_image_format()`
- ✅ `get_data_preprocessing()`

**Nicio schimbare necesară în template-urile TensorFlow existente!**

### Template PyTorch (NOU în v2)

**Funcții obligatorii pentru PyTorch:**
```python
# OBLIGATORIU: Antrenare cu PyTorch
def train_neural_network(model, train_loader, test_loader, epochs, verbose):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    # ... implementare antrenare
    return {'history': {...}}

# OBLIGATORIU: Încărcare model PyTorch
def load_model_config(filepath: str) -> nn.Module:
    model = torch.load(filepath, map_location='cpu')
    return model

# OPȚIONAL dar recomandat:
def get_model_weights(model: nn.Module) -> list:
    return [param.detach().cpu().clone() for param in model.parameters()]

def set_model_weights(model: nn.Module, weights: list):
    with torch.no_grad():
        for param, weight in zip(model.parameters(), weights):
            param.copy_(weight.to(param.device))
```

---

## ⚠️ Breaking Changes

### 1. Ordinea Parametrilor

**v1:**
```
test_file N M NN_NAME_PATH data_folder alternative_data R ROUNDS
```

**v2:**
```
task_dir model_name clean_data_folder poisoned_data_folder N M R ROUNDS
```

**Acțiune necesară:**
- 🔄 Reordonează parametrii în scripturile care apelează simulatorul
- 🔄 Actualizează orchestrator.py conform exemplelor

### 2. Path-uri Absolute → Relative

**v1:** Toate path-urile sunt absolute
**v2:** Doar `task_dir` este absolut, restul sunt relative

**Acțiune necesară:**
- ✅ Asigură-te că `task_dir` există și conține toate resursele
- ✅ Folosește nume simple pentru model și foldere date

### 3. Rezultate în Locație Fixă

**v1:** `test_file` poate fi oriunde
**v2:** Rezultatele ÎNTOTDEAUNA în `task_dir/results/`

**Acțiune necesară:**
- 🔄 Actualizează cod-ul care citește rezultatele
- ✅ Caută întotdeauna în `task_dir/results/<results_file>`

### 4. Logging în Fișier

**v1:** Logging doar în console
**v2:** Logging în console + `task_dir/results/simulation.log`

**Acțiune necesară:**
- ✅ Verifică spațiu pe disc pentru log files
- ✅ Implementează rotație log-uri dacă rulezi multe simulări

---

## 🚀 Avantaje Noi în v2

### 1. Suport Multi-Framework
```python
# TensorFlow
python fd_simulator_v2.py /task model.keras clean poisoned 10 2 2 5 --template tf_template.py

# PyTorch
python fd_simulator_v2.py /task model.pth clean poisoned 10 2 2 5 --template pytorch_template.py
```

### 2. Error Handling Îmbunătățit
```python
# v1: Crash direct
Exception: Model not found
# Script se oprește, dar celelalte thread-uri pot continua zombie

# v2: Graceful shutdown
CRITICAL ERROR: Model loading failed: [Errno 2] No such file or directory
  File "fd_simulator_v2.py", line 892, in _load_base_model
    model = tf.keras.models.load_model(self.model_path)
FileNotFoundError: [Errno 2] No such file or directory: 'model.keras'

ERROR: Client 0: Error detected, stopping simulation
ERROR: Client 1: Error detected, stopping simulation
...
✓ All threads stopped gracefully
Exit code: 1
```

### 3. Logging Detaliat
```bash
# v1: Doar console output
Starting simulation...
Client 0: Round 1 sent
Client 1: Round 1 sent

# v2: Log structurat în fișier + console
cat task_dir/results/simulation.log

2025-01-15 10:30:45 - [INFO] - ======================================================================
2025-01-15 10:30:45 - [INFO] - Simulation started at 2025-01-15 10:30:45
2025-01-15 10:30:45 - [INFO] - Task directory: /path/to/task_dir
2025-01-15 10:30:45 - [INFO] - Detected framework: TENSORFLOW
2025-01-15 10:30:45 - [INFO] - Server initialized:
2025-01-15 10:30:45 - [INFO] -   - Framework: TENSORFLOW
2025-01-15 10:30:45 - [INFO] -   - Clients: 10 (Malicious: 2)
...
```

### 4. Metrici Îmbunătățite
```json
// v1: Metrici de bază
{
  "final_accuracy": 0.8789,
  "round_metrics_history": [...]
}

// v2: Metrici complete + metadata
{
  "simulation_info": {
    "framework": "tensorflow",
    "timestamp": "2025-01-15T10:30:45",
    ...
  },
  "convergence_metrics": [0.7889, 0.8123, 0.8345, 0.8567, 0.8789],
  "weight_divergence": [0.0234, 0.0198, 0.0167, 0.0145, 0.0123],
  "round_times": [295.23, 287.45, 279.67, 271.89, 264.12],
  ...
}
```

---

## 📝 Checklist Migrare

### Pentru Developers

- [ ] Actualizează `orchestrator.py` cu noua interfață
- [ ] Modifică ordinea parametrilor în toate apelurile
- [ ] Schimbă path-uri absolute → relative
- [ ] Actualizează cod de citire rezultate (locație fixă)
- [ ] Testează cu TensorFlow models
- [ ] Testează cu PyTorch models (dacă e cazul)
- [ ] Verifică gestionarea erorilor
- [ ] Configurează rotație log-uri pentru producție

### Pentru Template Authors

#### TensorFlow Templates
- [ ] **Nicio modificare necesară!** ✅
- [ ] (Opțional) Adaugă mai multe funcții custom pentru control fin

#### PyTorch Templates
- [ ] Adaugă funcția `train_neural_network()` (OBLIGATORIU)
- [ ] Adaugă funcția `load_model_config()` (OBLIGATORIU)
- [ ] (Recomandat) Adaugă `get_model_weights()` și `set_model_weights()`
- [ ] (Recomandat) Adaugă `get_image_format()`
- [ ] Testează antrenarea cu noul simulator

### Pentru DevOps

- [ ] Instalează ambele framework-uri în mediul de producție
  ```bash
  pip install tensorflow torch torchvision --break-system-packages
  ```
- [ ] Verifică compatibilitatea versiunilor
  ```bash
  python -c "import tensorflow as tf; print(tf.__version__)"
  python -c "import torch; print(torch.__version__)"
  ```
- [ ] Configurează monitorizare pentru log files
- [ ] Setează limite de spațiu pentru logs
- [ ] Actualizează CI/CD pipeline cu noii parametri
- [ ] Testează recovery după erori

---

## 🐛 Troubleshooting Migrare

### Problemă 1: "FileNotFoundError: model.keras"

**Cauză:** Path-ul către model este incorect în v2

**Soluție v1:**
```bash
python fd_simulator.py ... /full/path/to/model.keras ...
```

**Soluție v2:**
```bash
python fd_simulator_v2.py /full/path/to/task_dir model.keras ...
# Model trebuie să fie în task_dir/model.keras
```

### Problemă 2: "Results file not found"

**Cauză:** Cauți rezultatele în locația greșită

**v1:**
```python
results_path = Path(args.test_file)  # Custom location
```

**v2:**
```python
results_path = Path(args.task_dir) / "results" / args.results_file  # Fixed location
```

### Problemă 3: "Template function not found"

**Cauză:** Template PyTorch lipsește funcții obligatorii

**Soluție:**
```python
# Adaugă în template:
def train_neural_network(model, train_loader, test_loader, epochs, verbose):
    # Implementare PyTorch training
    pass

def load_model_config(filepath):
    return torch.load(filepath, map_location='cpu')
```

### Problemă 4: "Wrong number of arguments"

**Cauză:** Ai uitat să actualizezi ordinea parametrilor

**v1 order:** `test_file N M model data alt_data R ROUNDS`
**v2 order:** `task_dir model clean_data poisoned_data N M R ROUNDS`

**Verifică:**
```bash
python fd_simulator_v2.py --help
```

---

## 📞 Support

Pentru probleme specifice migrării:

1. **Verifică documentația completă:** `FD_SIMULATOR_V2_DOCUMENTATION.md`
2. **Consultă exemplele:** 
   - TensorFlow: `template_code.py` (din uploads)
   - PyTorch: `pytorch_template_example.py`
3. **Verifică log-urile:** `task_dir/results/simulation.log`
4. **Testează într-un mediu de development** înainte de producție

---

## ✅ Concluzie

Migrarea de la v1 la v2 aduce:
- ✅ Simplitate în parametrizare
- ✅ Suport multi-framework
- ✅ Logging îmbunătățit
- ✅ Error handling robust
- ✅ Metrici mai complete

**Efort de migrare:**
- Orchestrator: ~30 minute (update parametri)
- Templates TensorFlow: 0 minute (compatibile direct)
- Templates PyTorch: ~2 ore (implementare funcții noi)
- Testing: ~1-2 ore

**ROI:**
- Mai puține bugs în producție
- Debug mai ușor cu logging complet
- Suport pentru mai multe framework-uri
- Cod mai curat și mentenabil
