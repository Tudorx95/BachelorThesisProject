#!/usr/bin/env python3
"""
Enhanced Federated Learning Simulator with JSON Metrics Storage
FRAMEWORK-AGNOSTIC VERSION (TensorFlow + PyTorch)

Modifications:
- JSON file path as argument for centralized test metrics
- Per-client metrics stored in test JSON file
- Thread-safe JSON file operations with FileLock
- Framework auto-detection from model file extension
- Conditional imports (TensorFlow OR PyTorch)
- Delegates to template_code.py for all framework-specific operations
"""

import sys
import os
import threading
import time
import queue
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
import argparse
import logging
import json
import glob
from collections import defaultdict
from pathlib import Path

try:
    from filelock import FileLock
except ImportError:
    print("WARNING: filelock not installed. Install with: pip install filelock")
    print("Falling back to basic threading lock (may not work across processes)")
    FileLock = None

# ============================================================================
# FRAMEWORK DETECTION & CONDITIONAL IMPORTS
# ============================================================================

def detect_framework_from_model(model_path):
    """Detectează framework-ul pe baza extensiei modelului"""
    if model_path.endswith('.keras'):
        return 'tensorflow'
    elif model_path.endswith('.pth'):
        return 'pytorch'
    else:
        raise ValueError(f"Unknown model format: {model_path}. Expected .keras or .pth")

# Parse model path EARLY pentru detectare framework
parser_temp = argparse.ArgumentParser(add_help=False)
parser_temp.add_argument('test_file', type=str)
parser_temp.add_argument('N', type=int)
parser_temp.add_argument('M', type=int)
parser_temp.add_argument('NN_NAME_PATH', type=str)
args_temp, _ = parser_temp.parse_known_args()

# Detectare framework
FRAMEWORK = detect_framework_from_model(args_temp.NN_NAME_PATH)
print(f"🔍 Detected framework: {FRAMEWORK.upper()}")

# Import conditional
if FRAMEWORK == 'tensorflow':
    print("📦 Loading TensorFlow...")
    import tensorflow as tf
    # Disable TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    DEVICE = None  # TensorFlow gestionează automat
else:  # pytorch
    print("📦 Loading PyTorch...")
    import torch
    import torch.nn as nn
    # Setare device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 PyTorch device: {DEVICE}")

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# JSON FILE MANAGER (Thread-safe)
# ============================================================================
class MetricsJSONManager:
    """Manager pentru citire/scriere thread-safe în fișierul JSON de metrici."""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.lock_path = json_path + '.lock'
        
        if FileLock:
            self.file_lock = FileLock(self.lock_path, timeout=30)
        else:
            self.file_lock = threading.Lock()
        
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Creează fișierul JSON dacă nu există."""
        os.makedirs(os.path.dirname(self.json_path) if os.path.dirname(self.json_path) else '.', exist_ok=True)
        if not os.path.exists(self.json_path):
            with open(self.json_path, 'w') as f:
                json.dump({}, f)
            logger.info(f"Created metrics JSON file: {self.json_path}")
    
    def read_metrics(self) -> dict:
        """Citește metricile din JSON (thread-safe)."""
        if FileLock:
            with self.file_lock:
                return self._read_file()
        else:
            with self.file_lock:
                return self._read_file()
    
    def _read_file(self) -> dict:
        """Internal method to read file."""
        try:
            with open(self.json_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"JSON decode error, returning empty dict")
            return {}
        except Exception as e:
            logger.error(f"Error reading metrics: {e}")
            return {}
    
    def write_metrics(self, data: dict):
        """Scrie metricile în JSON (thread-safe)."""
        if FileLock:
            with self.file_lock:
                self._write_file(data)
        else:
            with self.file_lock:
                self._write_file(data)
    
    def _write_file(self, data: dict):
        """Internal method to write file."""
        try:
            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=2, default=self._json_serializer)
            logger.debug(f"Metrics written to {self.json_path}")
        except Exception as e:
            logger.error(f"Error writing metrics: {e}")
    
    @staticmethod
    def _json_serializer(obj):
        """Serializare custom pentru tipuri numpy și torch."""
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # PyTorch tensors
        if FRAMEWORK == 'pytorch':
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
        raise TypeError(f"Type {type(obj)} not serializable")


# ============================================================================
# TEMPLATE FUNCTIONS LOADER
# ============================================================================
import importlib.util

class TemplateFunctions:
    def __init__(self):
        self.module = None
        self.available = False
    
    def load_template(self, template_path: str):
        try:
            spec = importlib.util.spec_from_file_location("user_template", template_path)
            self.module = importlib.util.module_from_spec(spec)
            sys.modules["user_template"] = self.module
            spec.loader.exec_module(self.module)
            self.available = True
            logger.info(f"✓ Template loaded: {template_path}")
            functions = [name for name in dir(self.module) 
                        if callable(getattr(self.module, name)) and not name.startswith('_')]
            logger.info(f"  Available functions: {', '.join(functions)}")
        except Exception as e:
            logger.error(f"✗ Error loading template: {e}")
            self.available = False
            raise
    
    def get_function(self, func_name: str):
        if not self.available:
            raise RuntimeError("Template not loaded")
        if not hasattr(self.module, func_name):
            raise AttributeError(f"Function '{func_name}' not found")
        return getattr(self.module, func_name)
    
    def has_function(self, func_name: str) -> bool:
        return self.available and hasattr(self.module, func_name)


TEMPLATE_FUNCS = TemplateFunctions()

# ============================================================================
# FRAMEWORK-AGNOSTIC HELPER FUNCTIONS
# ============================================================================

def move_model_to_device(model):
    """Move model to appropriate device (only for PyTorch)"""
    if FRAMEWORK == 'pytorch':
        return model.to(DEVICE)
    return model  # TensorFlow gestionează automat


def load_model_framework_agnostic(model_path, use_template=False):
    """Încarcă modelul folosind framework-ul potrivit"""
    if use_template and TEMPLATE_FUNCS.has_function('load_model_config'):
        load_func = TEMPLATE_FUNCS.get_function('load_model_config')
        model = load_func(model_path)
    else:
        if FRAMEWORK == 'tensorflow':
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:  # pytorch
            # Trebuie să importăm create_model din template
            if TEMPLATE_FUNCS.has_function('create_model'):
                create_model_func = TEMPLATE_FUNCS.get_function('create_model')
                model = create_model_func()
                
                checkpoint = torch.load(model_path, map_location=DEVICE)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                raise RuntimeError("PyTorch model loading requires create_model() in template_code.py")
    
    return move_model_to_device(model)


def get_model_weights_framework_agnostic(model, use_template=False):
    """Extrage weights folosind framework-ul potrivit"""
    if use_template and TEMPLATE_FUNCS.has_function('get_model_weights'):
        get_weights_func = TEMPLATE_FUNCS.get_function('get_model_weights')
        return get_weights_func(model)
    else:
        if FRAMEWORK == 'tensorflow':
            return model.get_weights()
        else:  # pytorch
            weights = []
            for param in model.parameters():
                weights.append(param.data.cpu().numpy())
            return weights


def set_model_weights_framework_agnostic(model, weights, use_template=False):
    """Setează weights folosind framework-ul potrivit"""
    if use_template and TEMPLATE_FUNCS.has_function('set_model_weights'):
        set_weights_func = TEMPLATE_FUNCS.get_function('set_model_weights')
        set_weights_func(model, weights)
    else:
        if FRAMEWORK == 'tensorflow':
            model.set_weights(weights)
        else:  # pytorch
            with torch.no_grad():
                for param, weight in zip(model.parameters(), weights):
                    param.data = torch.from_numpy(weight).to(param.device)


def get_model_output_shape(model):
    """Obține numărul de clase de output"""
    if FRAMEWORK == 'tensorflow':
        return model.output_shape[-1]
    else:  # pytorch
        # Presupunem că ultimul layer este Linear/fully connected
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        raise RuntimeError("Could not determine output shape for PyTorch model")


# ============================================================================
# ENHANCED FEDERATED SERVER
# ============================================================================
class EnhancedFederatedServer:
    def __init__(self, num_clients, num_malicious, nn_path, nn_name, data_folder, alternative_data,
                 rounds, r, strategy="first", data_poisoning=False, use_template=False,
                 test_json_path=None, data_poison_protection='fedavg'):
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.nn_path = nn_path
        self.nn_name = nn_name
        self.data_folder = data_folder
        self.alternative_data = alternative_data
        self.rounds = rounds
        self.R = r
        self.strategy = strategy
        self.data_poisoning = data_poisoning
        self.use_template = use_template
        self.data_poison_protection = data_poison_protection
        
        # JSON Metrics Manager
        self.test_json_path = test_json_path
        self.json_manager = MetricsJSONManager(test_json_path) if test_json_path else None
        
        # Sincronizare
        self.client_queues = {}
        self.server_queue = queue.Queue()
        
        # Creează queue-uri pentru clienți AICI (în __init__, nu în run())
        for i in range(num_clients):
            self.client_queues[i] = queue.Queue()
        
        # Încărcare model global
        logger.info(f"Loading global model: {nn_path} ({FRAMEWORK.upper()})")
        self.global_model = load_model_framework_agnostic(nn_path, use_template)
        self.global_weights = get_model_weights_framework_agnostic(self.global_model, use_template)
        
        # Metrici
        self.round_metrics_history = []
        self.convergence_metrics = []
        self.weight_divergence = []
        self.round_times = []
        self.malicious_clients = []
        
        # Identificare clienți malițioși
        self._assign_malicious_clients()
    
    def _assign_malicious_clients(self):
        """Atribuie ID-uri de clienți malițioși pe baza strategiei"""
        if self.num_malicious == 0:
            self.malicious_clients = []
            logger.info("No malicious clients")
            return
        
        if self.strategy == 'first':
            self.malicious_clients = list(range(self.num_malicious))
        elif self.strategy == 'last':
            self.malicious_clients = list(range(self.num_clients - self.num_malicious, self.num_clients))
        elif self.strategy in ['alternate', 'alternate_data']:
            self.malicious_clients = list(range(0, self.num_clients, 2))[:self.num_malicious]
        else:
            self.malicious_clients = list(range(self.num_malicious))
        
        logger.info(f"Malicious clients: {self.malicious_clients}")
    
    def _aggregate_weights_fedavg(self, client_weights, client_sizes):
        """FedAvg: weighted average based on dataset size"""
        total_size = sum(client_sizes)
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        
        for client_w, size in zip(client_weights, client_sizes):
            weight = size / total_size
            for i, w in enumerate(client_w):
                avg_weights[i] += w * weight
        
        return avg_weights
    
    def _aggregate_weights_krum(self, client_weights, num_malicious):
        """Krum: selects one client with smallest distance sum"""
        num_clients = len(client_weights)
        distances = np.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = sum(np.linalg.norm(w_i - w_j) for w_i, w_j in zip(client_weights[i], client_weights[j]))
                distances[i, j] = dist
                distances[j, i] = dist
        
        scores = []
        for i in range(num_clients):
            sorted_distances = np.sort(distances[i])
            score = np.sum(sorted_distances[:num_clients - num_malicious - 1])
            scores.append(score)
        
        selected_idx = np.argmin(scores)
        logger.info(f"Krum selected client {selected_idx}")
        return client_weights[selected_idx]
    
    def _aggregate_weights_trimmed_mean(self, client_weights, trim_ratio=0.1):
        """Trimmed Mean: remove top/bottom trim_ratio% and average"""
        num_clients = len(client_weights)
        num_trim = max(1, int(num_clients * trim_ratio))
        
        aggregated = []
        for layer_idx in range(len(client_weights[0])):
            layer_weights = [cw[layer_idx] for cw in client_weights]
            layer_weights_sorted = np.sort(layer_weights, axis=0)
            trimmed = layer_weights_sorted[num_trim:-num_trim] if num_trim > 0 else layer_weights_sorted
            aggregated.append(np.mean(trimmed, axis=0))
        
        return aggregated
    
    def _aggregate_weights_median(self, client_weights):
        """Median aggregation"""
        aggregated = []
        for layer_idx in range(len(client_weights[0])):
            layer_weights = [cw[layer_idx] for cw in client_weights]
            aggregated.append(np.median(layer_weights, axis=0))
        return aggregated
    
    def _aggregate_weights(self, client_weights, client_sizes):
        """Agregare ponderi cu protecție împotriva data poisoning"""
        method = self.data_poison_protection.lower()
        
        if method == 'krum':
            return self._aggregate_weights_krum(client_weights, self.num_malicious)
        elif method == 'trimmed_mean':
            return self._aggregate_weights_trimmed_mean(client_weights, trim_ratio=0.2)
        elif method == 'median':
            return self._aggregate_weights_median(client_weights)
        elif method == 'trimmed_mean_krum':
            trimmed = self._aggregate_weights_trimmed_mean(client_weights, trim_ratio=0.1)
            return trimmed
        elif method == 'random':
            idx = np.random.randint(0, len(client_weights))
            logger.info(f"Random aggregation: selected client {idx}")
            return client_weights[idx]
        else:  # fedavg (default)
            return self._aggregate_weights_fedavg(client_weights, client_sizes)
    
    def _evaluate_global_model(self):
        """Evaluare model global pe date de test"""
        try:
            if self.use_template and TEMPLATE_FUNCS.has_function('load_client_data'):
                load_data_func = TEMPLATE_FUNCS.get_function('load_client_data')
                _, test_ds = load_data_func(self.data_folder, batch_size=32)
            else:
                # Fallback la TensorFlow
                if FRAMEWORK == 'tensorflow':
                    test_ds = tf.keras.utils.image_dataset_from_directory(
                        os.path.join(self.data_folder, 'test'),
                        image_size=(32, 32),
                        batch_size=32,
                        shuffle=False
                    )
                else:
                    raise RuntimeError("PyTorch requires load_client_data() in template")
            
            if self.use_template and TEMPLATE_FUNCS.has_function('calculate_metrics'):
                calc_metrics_func = TEMPLATE_FUNCS.get_function('calculate_metrics')
                metrics = calc_metrics_func(self.global_model, test_ds)
                return metrics.get('accuracy', 0.0)
            else:
                # Evaluare manuală
                y_true, y_pred = [], []
                
                if FRAMEWORK == 'tensorflow':
                    for images, labels in test_ds:
                        predictions = self.global_model.predict(images, verbose=0)
                        y_pred.extend(np.argmax(predictions, axis=1))
                        y_true.extend(np.argmax(labels.numpy(), axis=1))
                else:  # pytorch
                    self.global_model.eval()
                    with torch.no_grad():
                        for images, labels in test_ds:
                            images = images.to(DEVICE)
                            outputs = self.global_model(images)
                            _, predicted = torch.max(outputs.data, 1)
                            
                            if len(labels.shape) > 1 and labels.shape[1] > 1:
                                labels = torch.argmax(labels, dim=1)
                            
                            y_true.extend(labels.cpu().numpy())
                            y_pred.extend(predicted.cpu().numpy())
                
                from sklearn.metrics import accuracy_score
                return accuracy_score(y_true, y_pred)
                
        except Exception as e:
            logger.error(f"Error evaluating global model: {e}")
            return 0.0
    
    def run(self):
        """Rulează simularea FL"""
        logger.info(f"Starting FL simulation with {self.num_clients} clients ({self.num_malicious} malicious)")
        logger.info(f"Framework: {FRAMEWORK.upper()}, Protection: {self.data_poison_protection}")
        
        # Distribuie weights inițiale
        logger.info("Distributing initial weights to clients...")
        for i in range(self.num_clients):
            self.client_queues[i].put({
                'type': 'base_weights',
                'weights': self.global_weights
            })
        
        # Așteaptă confirmări
        for i in range(self.num_clients):
            try:
                msg = self.server_queue.get(timeout=60)
                if msg['type'] != 'weights_received':
                    logger.error(f"Unexpected message from client {msg.get('client_id', '?')}")
            except queue.Empty:
                logger.error(f"Timeout waiting for client confirmations")
                return
        
        logger.info("All clients received base weights")
        
        # Rundă de antrenare
        for round_nr in range(self.rounds):
            round_start = time.time()
            logger.info(f"\n{'='*70}")
            logger.info(f"ROUND {round_nr + 1}/{self.rounds} ({FRAMEWORK.upper()})")
            logger.info(f"{'='*70}")
            
            round_updates = []
            
            # Colectează update-uri de la clienți
            for i in range(self.num_clients):
                try:
                    update = self.server_queue.get(timeout=600)
                    if update['type'] == 'round_update' and update['round'] == round_nr:
                        round_updates.append(update)
                except queue.Empty:
                    logger.error(f"Timeout waiting for client {i} in round {round_nr}")
            
            if len(round_updates) < self.num_clients:
                logger.warning(f"Only {len(round_updates)}/{self.num_clients} clients responded")
            
            # Agregare weights
            client_weights = [upd['weights'] for upd in round_updates]
            client_sizes = [1] * len(round_updates)  # Presupunem dimensiuni egale
            
            self.global_weights = self._aggregate_weights(client_weights, client_sizes)
            set_model_weights_framework_agnostic(self.global_model, self.global_weights, self.use_template)
            
            # Evaluare
            global_accuracy = self._evaluate_global_model()
            round_time = time.time() - round_start
            
            # Metrici per rundă
            round_metrics = {
                'round': round_nr,
                'accuracy': float(global_accuracy),
                'num_clients': len(round_updates),
                'round_time': float(round_time),
                'framework': FRAMEWORK
            }
            
            self.round_metrics_history.append(round_metrics)
            self.round_times.append(round_time)
            
            logger.info(f"Round {round_nr}: Accuracy = {global_accuracy:.4f}, Time = {round_time:.2f}s")
            
            # Distribuie weights actualizate
            if round_nr < self.rounds - 1:
                for i in range(self.num_clients):
                    self.client_queues[i].put({
                        'type': 'updated_weights',
                        'round': round_nr,
                        'weights': self.global_weights
                    })
                
                # Așteaptă confirmări
                for i in range(self.num_clients):
                    try:
                        self.server_queue.get(timeout=60)
                    except queue.Empty:
                        logger.warning(f"Timeout waiting for weight confirmation from client {i}")
        
        # Notifică sfârșit simulare
        for i in range(self.num_clients):
            self.client_queues[i].put({'type': 'simulation_end'})
        
        # Salvează rezultate
        self._save_results()
        logger.info("FL simulation completed!")
    
    def _save_results(self):
        """Salvează rezultatele finale"""
        if not self.json_manager:
            return
        
        final_accuracy = self.round_metrics_history[-1]['accuracy'] if self.round_metrics_history else 0.0
        
        results = {
            'final_accuracy': final_accuracy,
            'round_metrics_history': self.round_metrics_history,
            'convergence_metrics': self.convergence_metrics,
            'weight_divergence': self.weight_divergence,
            'round_times': self.round_times,
            'malicious_clients': self.malicious_clients,
            'framework': FRAMEWORK,
            'protection_method': self.data_poison_protection,
            'total_rounds': self.rounds,
            'num_clients': self.num_clients,
            'num_malicious': self.num_malicious
        }
        
        self.json_manager.write_metrics(results)
        logger.info(f"Results saved to {self.test_json_path}")


# ============================================================================
# ENHANCED FEDERATED CLIENT
# ============================================================================
class EnhancedFederatedClient:
    def __init__(self, client_id, server, data_folder, alternative_data, r, rounds, 
                 strategy, nn_path, use_template=False):
        self.client_id = client_id
        self.server = server
        self.data_folder = data_folder
        self.alternative_data = alternative_data
        self.R = r
        self.rounds = rounds
        self.strategy = strategy
        self.nn_path = nn_path
        self.use_template = use_template
        
        self.is_malicious = client_id in server.malicious_clients
        self.client_type = "malicious" if self.is_malicious else "honest"
        
        self.client_queue = server.client_queues[client_id]
        self.model = None
        self.current_weights = None
    
    def _get_data_path(self, round_nr):
        """Determină path-ul către date pentru rundă"""
        if self.is_malicious and round_nr < self.R:
            return self.alternative_data
        return self.data_folder
    
    def train_one_round(self, round_nr):
        """Antrenează modelul pentru o rundă"""
        try:
            data_path = self._get_data_path(round_nr)
            
            # Încarcă date
            if self.use_template and TEMPLATE_FUNCS.has_function('load_client_data'):
                load_data_func = TEMPLATE_FUNCS.get_function('load_client_data')
                train_ds, test_ds = load_data_func(data_path, batch_size=32)
            else:
                raise RuntimeError("load_client_data() required in template_code.py")
            
            # Setează weights curente
            set_model_weights_framework_agnostic(self.model, self.current_weights, self.use_template)
            
            # Antrenare
            if self.use_template and TEMPLATE_FUNCS.has_function('train_neural_network'):
                train_func = TEMPLATE_FUNCS.get_function('train_neural_network')
                train_func(self.model, train_ds, epochs=1, verbose=0)
            else:
                if FRAMEWORK == 'tensorflow':
                    self.model.fit(train_ds, epochs=1, verbose=0)
                else:  # pytorch
                    raise RuntimeError("train_neural_network() required in template_code.py for PyTorch")
            
            # Evaluare
            y_true, y_pred = [], []
            
            if FRAMEWORK == 'tensorflow':
                for images, labels in test_ds:
                    predictions = self.model.predict(images, verbose=0)
                    y_pred.extend(np.argmax(predictions, axis=1))
                    y_true.extend(np.argmax(labels.numpy(), axis=1))
            else:  # pytorch
                self.model.eval()
                with torch.no_grad():
                    for images, labels in test_ds:
                        images = images.to(DEVICE)
                        outputs = self.model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        if len(labels.shape) > 1 and labels.shape[1] > 1:
                            labels = torch.argmax(labels, dim=1)
                        
                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(predicted.cpu().numpy())
            
            # Calculează metrici
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_true, y_pred)
            
            if len(y_true) > 0:
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                precision = recall = f1 = 0.0
                
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error during training: {e}")
            import traceback
            traceback.print_exc()
            accuracy = precision = recall = f1 = 0.0
        
        # Extrage weights
        weights = get_model_weights_framework_agnostic(self.model, self.use_template)
        
        return weights, accuracy, precision, recall, f1
    
    def run(self):
        """Rulează client-ul"""
        poison_status = " [POISONED DATA]" if self.server.data_poisoning else ""
        logger.info(f"Client {self.client_id}: Starting {self.client_type} client{poison_status}")
        
        # Așteaptă weights inițiale
        base_weights_received = False
        while not base_weights_received:
            try:
                message = self.client_queue.get(timeout=300)
                if message['type'] == 'base_weights':
                    self.current_weights = message['weights']
                    logger.info(f"Client {self.client_id}: Received base weights")
                    self.server.server_queue.put({
                        'type': 'weights_received',
                        'client_id': self.client_id
                    })
                    base_weights_received = True
            except queue.Empty:
                logger.error(f"Client {self.client_id}: Timeout waiting for base weights")
                return
        
        # Încarcă model
        try:
            self.model = load_model_framework_agnostic(self.nn_path, self.use_template)
            self.current_num_classes = get_model_output_shape(self.model)
            logger.info(f"Client {self.client_id}: Model loaded ({FRAMEWORK.upper()}, classes: {self.current_num_classes})")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error loading model: {e}")
            return
        
        # Rundele de antrenare
        for round_nr in range(self.rounds):
            weights, accuracy, precision, recall, f1 = self.train_one_round(round_nr)
            
            if weights is not None:
                update = {
                    'type': 'round_update',
                    'client_id': self.client_id,
                    'round': round_nr,
                    'weights': weights,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'timestamp': time.time()
                }
                self.server.server_queue.put(update)
                
                client_type = "M" if self.is_malicious else "H"
                logger.info(f"[{client_type}] Client {self.client_id}: Round {round_nr} - Acc: {accuracy:.4f}")
            else:
                logger.error(f"Client {self.client_id}: Failed round {round_nr}")
            
            # Așteaptă weights actualizate pentru următoarea rundă
            if round_nr < self.rounds - 1:
                weights_received = False
                while not weights_received:
                    try:
                        message = self.client_queue.get(timeout=3000)
                        if message['type'] == 'updated_weights' and message['round'] == round_nr:
                            self.current_weights = message['weights']
                            self.server.server_queue.put({
                                'type': 'weights_received',
                                'client_id': self.client_id,
                                'round': round_nr
                            })
                            weights_received = True
                        elif message['type'] == 'simulation_end':
                            return
                    except queue.Empty:
                        logger.error(f"Client {self.client_id}: Timeout waiting for updated weights round {round_nr}")
                        return
        
        logger.info(f"Client {self.client_id}: Completed all rounds")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Federated Learning Simulator (TensorFlow + PyTorch)'
    )
    parser.add_argument('test_file', type=str, help='Path to test JSON file for storing metrics')
    parser.add_argument('N', type=int, help='Number of clients')
    parser.add_argument('M', type=int, help='Number of malicious clients')
    parser.add_argument('NN_NAME_PATH', type=str, help='Neural network path (.keras or .pth)')
    parser.add_argument('data_folder', type=str, help='Main data folder')
    parser.add_argument('alternative_data', type=str, help='Alternative data folder')
    parser.add_argument('R', type=int, help='Rounds using alternative data')
    parser.add_argument('ROUNDS', type=int, help='Total training rounds')
    parser.add_argument('--strategy', type=str, default='first',
                       choices=['first', 'last', 'alternate', 'alternate_data'],
                       help='Malicious client distribution strategy')
    parser.add_argument('--data_poisoning', action='store_true',
                       help='Enable data poisoning attack detection and logging')
    parser.add_argument('--data_poison_protection', type=str, default='fedavg',
                       choices=['fedavg', 'krum', 'trimmed_mean', 'median', 'trimmed_mean_krum', 'random'],
                       help='Aggregation method for data poison protection')
    parser.add_argument('--template', type=str, default=None,
                       help='Path to template_code.py for importing custom functions')
    
    args = parser.parse_args()
    
    # Validări
    if args.M > args.N or args.N <= 0 or args.ROUNDS <= 0:
        logger.error("Invalid arguments")
        return
    
    model_path = args.NN_NAME_PATH
    
    logger.info(f"Starting FL Simulator ({FRAMEWORK.upper()})")
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {args.data_folder}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
    
    if not os.path.exists(args.data_folder) or not os.path.exists(args.alternative_data):
        logger.error("Missing data folders")
        return
    
    # Încarcă template
    use_template = False
    if args.template and os.path.exists(args.template):
        try:
            TEMPLATE_FUNCS.load_template(args.template)
            use_template = True
            logger.info("✓ Template functions loaded")
        except Exception as e:
            logger.error(f"Could not load template: {e}")
            return
    else:
        # Caută template_code.py în directorul curent
        default_template = os.path.join(os.path.dirname(model_path), 'template_code.py')
        if os.path.exists(default_template):
            try:
                TEMPLATE_FUNCS.load_template(default_template)
                use_template = True
                logger.info(f"✓ Template loaded from: {default_template}")
            except Exception as e:
                logger.error(f"Could not load template: {e}")
                return
        else:
            logger.error(f"Template not found at: {default_template}")
            return
    
    poison_status = " with DATA POISONING" if args.data_poisoning else ""
    logger.info(f"Starting simulation{poison_status}")
    logger.info(f"N={args.N}, M={args.M}, Strategy={args.strategy}, Rounds={args.ROUNDS}")
    logger.info(f"Protection: {args.data_poison_protection}")
    logger.info(f"Results → {args.test_file}")
    
    model_name = Path(args.NN_NAME_PATH).stem
    
    # Creează server
    server = EnhancedFederatedServer(
        args.N, args.M, args.NN_NAME_PATH, model_name,
        args.data_folder, args.alternative_data,
        args.ROUNDS, args.R, args.strategy, args.data_poisoning,
        use_template, args.test_file, args.data_poison_protection
    )
    
    # Creează clienți
    clients = []
    client_threads = []
    
    for i in range(args.N):
        client = EnhancedFederatedClient(
            i, server, args.data_folder,
            args.alternative_data, args.R, 
            args.ROUNDS, args.strategy, model_path, use_template
        )
        clients.append(client)
        thread = threading.Thread(target=client.run, name=f"Client-{i}")
        thread.daemon = False
        client_threads.append(thread)
    
    # Start threads
    logger.info("Starting client threads...")
    for thread in client_threads:
        thread.start()
    
    time.sleep(2)
    
    # Run server
    try:
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()
    
    # Wait for clients
    logger.info("Waiting for clients to finish...")
    for i, thread in enumerate(client_threads):
        thread.join(timeout=120)
        if thread.is_alive():
            logger.warning(f"Client {i} still running")
    
    logger.info(f"✅ FL simulation completed! ({FRAMEWORK.upper()})")


if __name__ == "__main__":
    main()