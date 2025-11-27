#!/usr/bin/env python3
"""
Enhanced Federated Learning Simulator with JSON Metrics Storage - GPU Version
Modifications from CPU version:
- GPU memory growth and optimization enabled
- All functionalities maintained: aggregation, weight reset at client level
- Thread-safe JSON file operations with FileLock
"""

import sys
import os
import threading
import time
import queue
import numpy as np
import tensorflow as tf
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

# GPU Configuration
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    logging.info(f"GPUs detected by TensorFlow: {len(physical_devices)} → {[d.name for d in physical_devices]}")
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)
else:
    logging.warning("No GPU detected! Running on CPU (check CUDA_VISIBLE_DEVICES)")

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

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
        """Serializare custom pentru tipuri numpy."""
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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
# ENHANCED FEDERATED SERVER
# ============================================================================
class EnhancedFederatedServer:
    def __init__(self, num_clients, num_malicious, nn_path, nn_name, data_folder, alternative_data, 
                 rounds, r, strategy="first", data_poisoning=False, use_template=False, 
                 test_json_path=None):
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
        
        # JSON Metrics Manager
        self.test_json_path = test_json_path
        self.json_manager = MetricsJSONManager(test_json_path) if test_json_path else None
        
        # Sincronizare
        self.client_queues = {}
        self.server_queue = queue.Queue()
        self.round_lock = threading.Lock()
        self.clients_ready = threading.Event()
        
        # Model management
        self.base_weights = None
        self.current_weights = None
        self.round_data = {}
        self.weight_history = []
        
        # Enhanced tracking
        self.round_metrics_history = []
        self.convergence_metrics = []
        self.weight_divergence = []
        self.poisoning_info = {}
        self.round_times = []
        
        # File management
        self._load_existing_poisoning_info()
        self._extract_poisoning_info()
    
    
    def _load_existing_poisoning_info(self):
        """Încarcă informațiile despre data poisoning din fișierul test JSON dacă există."""
        if not self.data_poisoning or not self.json_manager:
            return
        
        try:
            existing_data = self.json_manager.read_metrics()
            
            # Caută informații relevante în JSON
            if 'attack_type' in existing_data:
                self.poisoning_info['test_file_info'] = {
                    'attack_type': existing_data.get('attack_type'),
                    'method': existing_data.get('method'),
                    'intensity': existing_data.get('intensity'),
                    'percentage': existing_data.get('percentage'),
                    'created_at': existing_data.get('created_at')
                }
                logger.info(f"Loaded poisoning info from test JSON: {self.test_json_path}")
        except Exception as e:
            logger.warning(f"Could not load existing poisoning info from test JSON: {e}")
    
    def _extract_poisoning_info(self):
        """Extrage informațiile despre data poisoning din fișierele *_attack_info.txt"""
        if not self.data_poisoning:
            return
        
        search_paths = [self.data_folder, self.alternative_data]
        
        for folder_path in search_paths:
            attack_files = glob.glob(os.path.join(folder_path, "*_attack_info.txt"))
            
            for attack_file in attack_files:
                try:
                    with open(attack_file, 'r') as f:
                        content = f.read()
                    
                    attack_info = {}
                    for line in content.strip().split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            attack_info[key.strip()] = value.strip()
                    
                    folder_name = os.path.basename(folder_path)
                    self.poisoning_info[folder_name] = {
                        'file_path': attack_file,
                        'info': attack_info
                    }
                    
                    logger.info(f"Data poisoning info extracted from {attack_file}")
                    
                except Exception as e:
                    logger.warning(f"Could not read attack info file {attack_file}: {e}")
    
    def _is_client_malicious(self, client_id):
        if self.strategy == "first":
            return client_id < self.num_malicious
        elif self.strategy == "last":
            return client_id >= (self.num_clients - self.num_malicious)
        elif self.strategy == "alternate":
            return client_id % 2 == 1 and client_id < self.num_malicious * 2
        return False
    
    def _calculate_weight_divergence(self, weights_list):
        if len(weights_list) < 2:
            return 0.0
        divergence = 0.0
        for i in range(len(weights_list)):
            for j in range(i + 1, len(weights_list)):
                layer_div = 0.0
                for layer_idx in range(len(weights_list[i])):
                    diff = weights_list[i][layer_idx] - weights_list[j][layer_idx]
                    layer_div += np.sum(diff ** 2)
                divergence += np.sqrt(layer_div)
        return divergence / (len(weights_list) * (len(weights_list) - 1) / 2)
    
    def _calculate_convergence_metrics(self, round_nr):
        if round_nr not in self.round_data or not self.round_data[round_nr]:
            return {"accuracy_std": 0.0, "accuracy_range": 0.0, "improving_clients": 0, "mean_accuracy": 0.0}
        current_accuracies = [u['accuracy'] for u in self.round_data[round_nr]]
        accuracy_std = np.std(current_accuracies)
        accuracy_range = np.max(current_accuracies) - np.min(current_accuracies)
        improving_clients = 0
        if round_nr > 0:
            prev_accuracies = [u['accuracy'] for u in self.round_data[round_nr - 1]]
            improving_clients = sum(1 for curr, prev in zip(current_accuracies, prev_accuracies) if curr > prev)
        return {
            "accuracy_std": float(accuracy_std),
            "accuracy_range": float(accuracy_range),
            "improving_clients": int(improving_clients),
            "mean_accuracy": float(np.mean(current_accuracies)),
            "accuracy_trend": float(np.mean(current_accuracies) - np.mean(prev_accuracies)) if round_nr > 0 else 0.0
        }
    
    def load_base_model(self):
        try:
            model_path = self.nn_path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found")
            logger.info(f"Loading base model from {model_path}")
            if self.use_template and TEMPLATE_FUNCS.has_function('load_model_config'):
                load_func = TEMPLATE_FUNCS.get_function('load_model_config')
                model = load_func(model_path.replace('.keras', ''))
            else:
                model = tf.keras.models.load_model(model_path, compile=False)
            if self.use_template and TEMPLATE_FUNCS.has_function('get_model_weights'):
                get_weights_func = TEMPLATE_FUNCS.get_function('get_model_weights')
                self.base_weights = get_weights_func(model)
            else:
                self.base_weights = model.get_weights()
            self.current_weights = [w.copy() for w in self.base_weights]
            del model
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def register_client(self, client_id):
        self.client_queues[client_id] = queue.Queue()
        client_type = "malicious" if self._is_client_malicious(client_id) else "honest"
        logger.info(f"Client {client_id} registered ({client_type})")
        if len(self.client_queues) == self.num_clients:
            self.clients_ready.set()
    
    def send_base_weights(self):
        if self.base_weights is None:
            logger.error("No base weights available")
            return False
        for client_id in self.client_queues:
            self.client_queues[client_id].put({
                'type': 'base_weights',
                'weights': [w.copy() for w in self.base_weights],
                'timestamp': time.time()
            })
        logger.info(f"Base weights sent to {len(self.client_queues)} clients")
        confirmations = 0
        timeout = 300
        start_time = time.time()
        while confirmations < self.num_clients and (time.time() - start_time) < timeout:
            try:
                message = self.server_queue.get(timeout=10)
                if message.get('type') == 'weights_received':
                    confirmations += 1
                    client_id = message['client_id']
                    logger.info(f"Client {client_id} confirmed weights received ({confirmations}/{self.num_clients})")
            except queue.Empty:
                continue
        if confirmations == self.num_clients:
            logger.info("All clients ready for training")
            return True
        else:
            logger.error(f"Only {confirmations}/{self.num_clients} clients confirmed. Aborting.")
            return False
    
    def wait_for_round_updates(self, round_nr):
        logger.info(f"Waiting for round {round_nr} updates from {self.num_clients} clients")
        received_updates = 0
        start_time = time.time()
        last_log_time = start_time
        round_metrics = {
            'round': round_nr,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        while received_updates < self.num_clients:
            try:
                update = self.server_queue.get(timeout=300)
                if update.get('type') == 'round_update' and update['round'] == round_nr:
                    self.round_data[round_nr].append(update)
                    received_updates += 1
                    client_id = update['client_id']
                    
                    # Collect metrics for averaging
                    round_metrics['accuracy'].append(float(update['accuracy']))
                    round_metrics['precision'].append(float(update['precision']))
                    round_metrics['recall'].append(float(update['recall']))
                    round_metrics['f1_score'].append(float(update['f1_score']))
                    
                    elapsed_time = time.time() - start_time
                    client_type = "M" if self._is_client_malicious(client_id) else "H"
                    logger.info(f"[{client_type}] Client {client_id} round {round_nr} "
                                f"({received_updates}/{self.num_clients}) - "
                                f"Acc: {update['accuracy']:.4f} - {elapsed_time:.1f}s")
                else:
                    logger.warning(f"Unexpected message: {update.get('type')} for round {update.get('round')}")
            except queue.Empty:
                current_time = time.time()
                if current_time - last_log_time >= 60:
                    elapsed_time = current_time - start_time
                    logger.info(f"Still waiting for round {round_nr} updates... ({received_updates}/{self.num_clients}) - {elapsed_time:.1f}s")
                    last_log_time = current_time
                continue
        # Calculate mean metrics for the round
        if round_metrics['accuracy']:
            mean_metrics = {
                'round': round_nr,
                'accuracy': float(np.mean(round_metrics['accuracy'])),
                'precision': float(np.mean(round_metrics['precision'])),
                'recall': float(np.mean(round_metrics['recall'])),
                'f1_score': float(np.mean(round_metrics['f1_score']))
            }
            self.round_metrics_history.append(mean_metrics)
        
        total_time = time.time() - start_time
        logger.info(f"All updates received for round {round_nr} in {total_time:.2f}s")
        return True
    
    

    def _aggregate_metrics(self, client_updates, aggregation_method='mean'):
        """
        Agregă metricile de la clienți folosind diferite metode.
        
        Args:
            client_updates: Lista cu update-uri de la clienți
            aggregation_method: Metoda de agregare ('mean', 'median', 'weighted_mean')
        
        Returns:
            Dictionary cu metricile agregate
        """
        try:
            if not client_updates:
                logger.error("No client updates to aggregate metrics from")
                return None
            
            # Extrage metricile de la toți clienții
            accuracies = [u['accuracy'] for u in client_updates]
            precisions = [u['precision'] for u in client_updates]
            recalls = [u['recall'] for u in client_updates]
            f1_scores = [u['f1_score'] for u in client_updates]
            
            # Agregare bazată pe metodă
            if aggregation_method == 'mean':
                aggregated_metrics = {
                    'accuracy': float(np.mean(accuracies)),
                    'precision': float(np.mean(precisions)),
                    'recall': float(np.mean(recalls)),
                    'f1_score': float(np.mean(f1_scores)),
                    'std_accuracy': float(np.std(accuracies)),
                    'std_precision': float(np.std(precisions)),
                    'std_recall': float(np.std(recalls)),
                    'std_f1_score': float(np.std(f1_scores)),
                    'min_accuracy': float(np.min(accuracies)),
                    'max_accuracy': float(np.max(accuracies))
                }
            
            elif aggregation_method == 'median':
                aggregated_metrics = {
                    'accuracy': float(np.median(accuracies)),
                    'precision': float(np.median(precisions)),
                    'recall': float(np.median(recalls)),
                    'f1_score': float(np.median(f1_scores)),
                    'std_accuracy': float(np.std(accuracies)),
                    'std_precision': float(np.std(precisions)),
                    'std_recall': float(np.std(recalls)),
                    'std_f1_score': float(np.std(f1_scores)),
                    'min_accuracy': float(np.min(accuracies)),
                    'max_accuracy': float(np.max(accuracies))
                }
            
            elif aggregation_method == 'weighted_mean':
                # Pentru implementare viitoare: se poate pondera după dimensiunea datasetului
                # Deocamdată, folosim media simplă
                aggregated_metrics = {
                    'accuracy': float(np.mean(accuracies)),
                    'precision': float(np.mean(precisions)),
                    'recall': float(np.mean(recalls)),
                    'f1_score': float(np.mean(f1_scores)),
                    'std_accuracy': float(np.std(accuracies)),
                    'std_precision': float(np.std(precisions)),
                    'std_recall': float(np.std(recalls)),
                    'std_f1_score': float(np.std(f1_scores)),
                    'min_accuracy': float(np.min(accuracies)),
                    'max_accuracy': float(np.max(accuracies))
                }
            
            else:
                logger.warning(f"Unknown aggregation method: {aggregation_method}, using mean")
                aggregated_metrics = {
                    'accuracy': float(np.mean(accuracies)),
                    'precision': float(np.mean(precisions)),
                    'recall': float(np.mean(recalls)),
                    'f1_score': float(np.mean(f1_scores)),
                    'std_accuracy': float(np.std(accuracies)),
                    'std_precision': float(np.std(precisions)),
                    'std_recall': float(np.std(recalls)),
                    'std_f1_score': float(np.std(f1_scores)),
                    'min_accuracy': float(np.min(accuracies)),
                    'max_accuracy': float(np.max(accuracies))
                }
            
            return aggregated_metrics
            
        except Exception as e:
            logger.error(f"Metrics aggregation failed: {e}")
            return None
    
    def _aggregate_weights(self, client_updates):
        """
        Agregă ponderile de la clienți folosind FedAvg (medie simplă).
        
        Args:
            client_updates: Lista cu update-uri de la clienți
        
        Returns:
            Lista cu ponderile agregate sau None în caz de eroare
        """
        try:
            if not client_updates:
                logger.error("No client updates to aggregate weights from")
                return None
            
            # Extrage ponderile
            weights_list = [update['weights'] for update in client_updates]
            
            # FedAvg: media simplă a ponderilor
            aggregated_weights = []
            for i in range(len(weights_list[0])):
                layer_weights = np.mean([w[i] for w in weights_list], axis=0)
                aggregated_weights.append(layer_weights)
            
            return aggregated_weights
            
        except Exception as e:
            logger.error(f"Weight aggregation failed: {e}")
            return None
    
    def aggregate_round(self, round_nr):
        """
        Agregă un round de antrenament - apelează funcțiile separate pentru metrici și ponderi.
        """
        if round_nr not in self.round_data or len(self.round_data[round_nr]) != self.num_clients:
            logger.error(f"Cannot aggregate round {round_nr} - incomplete data")
            return False
        
        with self.round_lock:
            start_time = time.time()
            updates = self.round_data[round_nr]
            
            # Calculează divergența ponderilor
            weights_list = [update['weights'] for update in updates]
            divergence = self._calculate_weight_divergence(weights_list)
            self.weight_divergence.append(float(divergence))
            
            # AGREGARE PONDERI - funcție separată
            aggregated_weights = self._aggregate_weights(updates)
            if aggregated_weights is None:
                logger.error(f"Failed to aggregate weights for round {round_nr}")
                return False
            
            self.current_weights = aggregated_weights
            self.weight_history.append([w.copy() for w in aggregated_weights])
            
            # Calculează metrici de convergență
            convergence = self._calculate_convergence_metrics(round_nr)
            self.convergence_metrics.append(convergence)
            
            aggregation_time = time.time() - start_time
            self.round_times.append(float(aggregation_time))
            
            # Separă update-urile în honest și malicious pentru logging
            honest_updates = [u for u in updates if not self._is_client_malicious(u['client_id'])]
            malicious_updates = [u for u in updates if self._is_client_malicious(u['client_id'])]
            
            # AGREGARE METRICI - funcție separată (mean implicit)
            # Poți schimba metoda aici: 'mean', 'median', 'weighted_mean'
            aggregation_method = 'mean'  # Schimbă acest parametru pentru testare manuală
            
            overall_metrics = self._aggregate_metrics(updates, aggregation_method)
            honest_metrics = self._aggregate_metrics(honest_updates, aggregation_method) if honest_updates else None
            malicious_metrics = self._aggregate_metrics(malicious_updates, aggregation_method) if malicious_updates else None
            
            # Pentru compatibilitate cu codul existent, folosește valorile din overall_metrics
            avg_accuracy = overall_metrics['accuracy'] if overall_metrics else 0.0
            honest_acc = honest_metrics['accuracy'] if honest_metrics else 0.0
            malicious_acc = malicious_metrics['accuracy'] if malicious_metrics else 0.0
            
            poison_status = " [POISONED]" if self.data_poisoning else ""
            logger.info(f"Round {round_nr} aggregated in {aggregation_time:.2f}s{poison_status} (method: {aggregation_method})")
            logger.info(f"  Overall: Acc={avg_accuracy:.4f}, Prec={overall_metrics['precision']:.4f}, "
                       f"Rec={overall_metrics['recall']:.4f}, F1={overall_metrics['f1_score']:.4f}")
            logger.info(f"  Honest: {honest_acc:.4f}, Malicious: {malicious_acc:.4f}")
            logger.info(f"  Divergence: {divergence:.4f}, Std: {convergence['accuracy_std']:.4f}")
            
            return True
    
    def send_updated_weights(self, round_nr):
        logger.info(f"Sending updated weights for round {round_nr}")
        for client_id in self.client_queues:
            self.client_queues[client_id].put({
                'type': 'updated_weights',
                'weights': [w.copy() for w in self.current_weights],
                'round': round_nr,
                'timestamp': time.time()
            })
        confirmations = 0
        timeout = 120
        start_time = time.time()
        while confirmations < self.num_clients and (time.time() - start_time) < timeout:
            try:
                message = self.server_queue.get(timeout=10)
                if message.get('type') == 'weights_received' and message.get('round') == round_nr:
                    confirmations += 1
            except queue.Empty:
                continue
        return confirmations == self.num_clients
    
    def save_to_test_json(self):
        """Salvează toate metricile în fișierul test JSON în formatul specificat."""
        if not self.json_manager:
            logger.warning("No test JSON manager configured")
            return
        
        # Construiește obiectul final
        results = {
            'strategy': self.strategy,
            'data_poisoning': self.data_poisoning,
            'poisoning_info': self.poisoning_info,
            'convergence_metrics': self.convergence_metrics,
            'weight_divergence': self.weight_divergence,
            'round_metrics_history': self.round_metrics_history,
            'round_times': self.round_times,
            'malicious_clients': [i for i in range(self.num_clients) if self._is_client_malicious(i)]
        }
        
        self.json_manager.write_metrics(results)
        logger.info(f"Results saved to test JSON: {self.test_json_path}")
    
    
    def run(self):
        poison_status = " with DATA POISONING" if self.data_poisoning else ""
        logger.info(f"Starting enhanced federated server{poison_status} with strategy: {self.strategy}")
        total_start = time.time()
        if not self.load_base_model():
            return
        logger.info(f"Waiting for {self.num_clients} clients to register...")
        if not self.clients_ready.wait(timeout=120):
            logger.error("Timeout waiting for clients to register")
            return
        if not self.send_base_weights():
            return
        
        for round_nr in range(self.rounds):
            logger.info(f"\n{'='*50}")
            logger.info(f"ROUND {round_nr} - Strategy: {self.strategy}{poison_status}")
            logger.info(f"{'='*50}")
            self.round_data[round_nr] = []
            if not self.wait_for_round_updates(round_nr):
                break
            if not self.aggregate_round(round_nr):
                break
            if round_nr < self.rounds - 1:
                if not self.send_updated_weights(round_nr):
                    logger.warning(f"Not all clients confirmed weights for round {round_nr}")
        for client_id in self.client_queues:
            self.client_queues[client_id].put({'type': 'simulation_end'})
        total_time = time.time() - total_start
        
        # Salvează în test JSON
        self.save_to_test_json()

        # Call save_model_config if available in template
        if self.use_template and TEMPLATE_FUNCS.has_function('save_model_config'):
            try:
                logger.info("Attempting to save model configuration...")
                save_model_func = TEMPLATE_FUNCS.get_function('save_model_config')
                # Create a temporary model to set weights
                model_path = f"{self.nn_name}.keras"
                logger.info(f"Loading model from: {model_path}")

                if TEMPLATE_FUNCS.has_function('load_model_config'):
                    load_func = TEMPLATE_FUNCS.get_function('load_model_config')
                    model = load_func(model_path.replace('.keras', ''))
                else:
                    model = tf.keras.models.load_model(model_path, compile=False)
                
                logger.info(f"Setting weights to model...")
                if TEMPLATE_FUNCS.has_function('set_model_weights'):
                    set_weights_func = TEMPLATE_FUNCS.get_function('set_model_weights')
                    set_weights_func(model, self.current_weights)
                else:
                    model.set_weights(self.current_weights)
                # Call save_model_config with model and output path
                output_path = f"{self.nn_name}_final.keras"
                logger.info(f"Saving model to: {output_path}")
                save_model_func(model, output_path)
                logger.info(f"Model configuration saved via template to {output_path}")
                del model
            except Exception as e:
                logger.error(f"Error calling save_model_config: {e}")
        
        logger.info(f"\nSimulation completed in {total_time:.2f}s")
        logger.info(f"Strategy: {self.strategy}")
        logger.info(f"Data Poisoning: {'Yes' if self.data_poisoning else 'No'}")
        
        if self.test_json_path:
            logger.info(f"Test Metrics JSON: {self.test_json_path}")


# ============================================================================
# ENHANCED FEDERATED CLIENT
# ============================================================================
class EnhancedFederatedClient:
    def __init__(self, client_id, server, data_folder, alternative_data, R, rounds, strategy, nn_path, use_template=False):
        self.client_id = client_id
        self.server = server
        self.is_malicious = self._determine_malicious_status(strategy)
        self.data_folder = data_folder
        self.alternative_data = alternative_data
        self.R = R
        self.rounds = rounds
        self.strategy = strategy
        self.model = None
        self.current_weights = None
        self.use_template = use_template
        self.current_num_classes = None
        self.original_num_classes = None
        self.base_model = None
        self.client_type = "malicious" if self.is_malicious else "honest"
        logger.info(f"Client {client_id} initialized as {self.client_type} (strategy: {strategy})")
        self.server.register_client(client_id)
        self.client_queue = self.server.client_queues[client_id]
        self.nn_path = nn_path
    
    def _determine_malicious_status(self, strategy):
        if strategy == "first":
            return self.client_id < self.server.num_malicious
        elif strategy == "last":
            return self.client_id >= (self.server.num_clients - self.server.num_malicious)
        elif strategy == "alternate":
            return self.client_id % 2 == 1 and self.client_id < self.server.num_malicious * 2
        return False
    
    def _get_data_path(self, round_nr):
        if not self.is_malicious:
            return os.path.join(self.data_folder, "train")
        if self.strategy == "alternate_data":
            if round_nr % 2 == 0:
                return os.path.join(self.alternative_data, "train")
            else:
                return os.path.join(self.data_folder, "train")
        else:
            # daca R == -1 atunci antreneaza doar cu poison data
            # altfel antreneaza primele runde doar cu poison data si apoi doar cu clean data
            if self.R == -1 or self.R > 0: # round_nr < self.R:
                self.R -= 1 
                return os.path.join(self.alternative_data, "train")
            else:
                return os.path.join(self.data_folder, "train")
    
    def load_data(self, data_path, round_nr):
        try:
            # Setează dimensiunea imaginii bazată pe model
            input_shape = self.model.input_shape[1:3]  # (height, width)
            expected_channels = self.model.input_shape[-1]
            
            logger.info(f"Client {self.client_id}: Model expects input shape: {input_shape}")

            # Determină formatul etichetelor bazat pe funcția de loss din template
            label_mode = "int"  # default pentru sparse_categorical_crossentropy
            
            if self.use_template:
                # Verifică dacă template-ul are funcție pentru a determina tipul de loss
                if TEMPLATE_FUNCS.has_function('get_loss_type'):
                    get_loss_func = TEMPLATE_FUNCS.get_function('get_loss_type')
                    loss_type = get_loss_func()
                    if loss_type == 'categorical_crossentropy':
                        label_mode = "categorical"
                    logger.info(f"Client {self.client_id}: Using loss type from template: {loss_type}")
                # Alternativ, verifică dacă modelul este deja compilat cu categorical_crossentropy
                elif hasattr(self.model, 'loss') and 'categorical' in str(self.model.loss).lower():
                    label_mode = "categorical"
                    logger.info(f"Client {self.client_id}: Detected categorical loss from model: {self.model.loss}")

            logger.info(f"Client {self.client_id}: Using label_mode: {label_mode}")
            
            color_mode = 'grayscale' if expected_channels == 1 else 'rgb'

            # Încarcă dataset-urile
            train_ds = tf.keras.utils.image_dataset_from_directory(
                data_path, image_size=input_shape, batch_size=32,
                seed=round_nr + self.client_id, validation_split=0.2,
                subset="training", label_mode=label_mode, shuffle=True, color_mode=color_mode)
            val_ds = tf.keras.utils.image_dataset_from_directory(
                data_path, image_size=input_shape, batch_size=32,
                seed=round_nr + self.client_id, validation_split=0.2,
                subset="validation", label_mode=label_mode, shuffle=False, color_mode=color_mode)

            # Funcție de preprocesare generică
            def generic_preprocess(image, label):
                # Normalizează
                image = tf.cast(image, tf.float32) / 255.0
                 
                # Asigură-te că imaginea are numărul corect de canale
                if expected_channels == 1 and image.shape[-1] == 3:
                    # Converteste RGB la grayscale dacă modelul așteaptă 1 canal
                    image = tf.image.rgb_to_grayscale(image)
                elif expected_channels == 3 and image.shape[-1] == 1:
                    # Converteste grayscale la RGB dacă modelul așteaptă 3 canale
                    image = tf.image.grayscale_to_rgb(image)
                return image, label

            # Aplică preprocesarea
            train_ds = train_ds.map(generic_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(generic_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

            # Optimizează
            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
            val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

            logger.info(f"Client {self.client_id}: Data loading successful (label_mode: {label_mode})")
            return train_ds, val_ds
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error loading data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def train_one_round(self, round_nr):
        data_path = self._get_data_path(round_nr)
        data_type = "poisoned" if "poisoned" in data_path else "normal"
        poison_indicator = " [POISONED]" if self.server.data_poisoning and data_type == "poisoned" else ""
        logger.info(f"Client {self.client_id} ({'M' if self.is_malicious else 'H'}): Round {round_nr} - {data_type} data{poison_indicator}")
        data = self.load_data(data_path, round_nr)
        if data is None:
            logger.error(f"Client {self.client_id}: Failed to load data for round {round_nr}")
            return None, 0.0, 0.0, 0.0, 0.0
        train_ds, val_ds = data
        
        # Setează ponderile - CRITICAL: Resetare weights la fiecare rundă
        if self.current_weights is not None:
            try:
                if self.use_template and TEMPLATE_FUNCS.has_function('set_model_weights'):
                    set_weights_func = TEMPLATE_FUNCS.get_function('set_model_weights')
                    set_weights_func(self.model, self.current_weights)
                else:
                    self.model.set_weights(self.current_weights)
            except Exception as e:
                logger.warning(f"Client {self.client_id}: Could not set weights: {e}")
        
        # Compilează modelul folosind funcția din template sau compilare implicită
        if not hasattr(self.model, '_is_compiled') or not self.model._is_compiled:
            if self.use_template and TEMPLATE_FUNCS.has_function('_model_compile'):
                try:
                    compile_func = TEMPLATE_FUNCS.get_function('_model_compile')
                    self.model = compile_func(self.model)
                    logger.info(f"Client {self.client_id}: Model compiled using template _model_compile")
                except Exception as e:
                    logger.error(f"Client {self.client_id}: Error compiling model with template function: {e}")
                    # Fallback la compilare implicită
                    self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            else:
                # Compilare implicită
                self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                logger.info(f"Client {self.client_id}: Model compiled with default settings")
        
        # Antrenare
        try:
            start_time = time.time()
            history = self.model.fit(train_ds, epochs=1, verbose=0)
            train_time = time.time() - start_time
            if self.is_malicious:
                logger.info(f"Client {self.client_id} (M): Training completed in {train_time:.2f}s with {data_type} data{poison_indicator}")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error during training: {e}")
            return None, 0.0, 0.0, 0.0, 0.0
        
        # Evaluare cu calcul de metrici complete
        try:
            loss, accuracy = self.model.evaluate(val_ds, verbose=0)
            y_true = []
            y_pred = []
            
            for batch_x, batch_y in val_ds:
                predictions = self.model.predict(batch_x, verbose=0)
                y_pred.extend(np.argmax(predictions, axis=1))
                
                # Gestionează diferite formate de etichete
                if len(batch_y.shape) > 1 and batch_y.shape[-1] > 1:
                    # Etichete one-hot - convertește la întreg
                    y_true.extend(np.argmax(batch_y.numpy(), axis=1))
                else:
                    # Etichete întregi - folosește direct
                    y_true.extend(batch_y.numpy().flatten())
            
            if len(y_true) > 0:
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                precision = recall = f1 = 0.0
                
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error during evaluation: {e}")
            accuracy = precision = recall = f1 = 0.0
        
        # Extrage ponderile
        if self.use_template and TEMPLATE_FUNCS.has_function('get_model_weights'):
            get_weights_func = TEMPLATE_FUNCS.get_function('get_model_weights')
            weights = get_weights_func(self.model)
        else:
            weights = self.model.get_weights()
            
        return weights, accuracy, precision, recall, f1

    def run(self):
        poison_status = " [POISONED DATA]" if self.server.data_poisoning else ""
        logger.info(f"Client {self.client_id}: Starting {self.client_type} client{poison_status} (strategy: {self.strategy})")
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
        try:
            model_path = self.nn_path
            if self.use_template and TEMPLATE_FUNCS.has_function('load_model_config'):
                load_func = TEMPLATE_FUNCS.get_function('load_model_config')
                self.model = load_func(model_path)
            else:
                self.model = tf.keras.models.load_model(model_path, compile=False)
            self.current_num_classes = self.model.output_shape[-1]
            self.original_num_classes = self.current_num_classes
            if self.use_template and TEMPLATE_FUNCS.has_function('set_model_weights'):
                set_weights_func = TEMPLATE_FUNCS.get_function('set_model_weights')
                set_weights_func(self.model, self.current_weights)
            else:
                self.model.set_weights(self.current_weights)
            logger.info(f"Client {self.client_id}: Model loaded (classes: {self.current_num_classes})")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error loading model: {e}")
            return
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
                logger.info(f"[{client_type}] Client {self.client_id}: Round {round_nr} sent - Acc: {accuracy:.4f}")
            else:
                logger.error(f"Client {self.client_id}: Failed to train round {round_nr}")
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


def main():
    parser = argparse.ArgumentParser(description='Enhanced Federated Learning Simulator with JSON Metrics Storage - GPU Version')
    parser.add_argument('test_file', type=str, help='Path to test JSON file for storing metrics')
    parser.add_argument('N', type=int, help='Number of clients')
    parser.add_argument('M', type=int, help='Number of malicious clients')
    parser.add_argument('NN_NAME_PATH', type=str, help='Neural network name')
    parser.add_argument('data_folder', type=str, help='Main data folder')
    parser.add_argument('alternative_data', type=str, help='Alternative data folder')
    parser.add_argument('R', type=int, help='Rounds using alternative data')    # important only for _get_data_path (choosing the dataset for that round)
    parser.add_argument('ROUNDS', type=int, help='Total training rounds')
    parser.add_argument('--strategy', type=str, default='first', 
                       choices=['first', 'last', 'alternate', 'alternate_data'],
                       help='Malicious client distribution strategy')
    parser.add_argument('--data_poisoning', action='store_true', 
                       help='Enable data poisoning attack detection and logging')
    parser.add_argument('--template', type=str, default=None,
                       help='Path to template_code.py for importing custom functions')
    
    args = parser.parse_args()
    
    if args.M > args.N or args.N <= 0 or args.ROUNDS <= 0:
        logger.error("Invalid arguments")
        return
    
    model_path =  f"{args.NN_NAME_PATH}"
    
    logger.info(f"File Path {Path(__file__)} and data_folder: {args.data_folder}; and model_path= {model_path}")
    if not os.path.exists(model_path) or not os.path.exists(args.data_folder) or not os.path.exists(args.alternative_data):
        logger.error("Missing files/folders")
        return
    
    use_template = False
    if args.template and os.path.exists(args.template):
        try:
            TEMPLATE_FUNCS.load_template(args.template)
            use_template = True
            logger.info("Template functions will be used")
        except Exception as e:
            logger.warning(f"Could not load template, using default functions: {e}")
            use_template = False
    
    poison_status = " with DATA POISONING detection" if args.data_poisoning else ""
    logger.info(f"Starting enhanced simulation{poison_status}: N={args.N}, M={args.M}, Strategy={args.strategy}, Rounds={args.ROUNDS}")
    logger.info(f"Test metrics will be saved to: {args.test_file}")
    
    model_name = Path(args.NN_NAME_PATH).stem   
    print(f"Model_Name: {model_name}") 
    server = EnhancedFederatedServer(args.N, args.M, args.NN_NAME_PATH, model_name, 
                                    args.data_folder, args.alternative_data, 
                                    args.ROUNDS, args.R, args.strategy, args.data_poisoning,
                                    use_template, args.test_file)
    
    clients = []
    client_threads = []
    
    for i in range(args.N):
        client = EnhancedFederatedClient(i, server, args.data_folder,
                                        args.alternative_data, args.R, 
                                        args.ROUNDS, args.strategy, model_path, use_template)
        clients.append(client)
        thread = threading.Thread(target=client.run, name=f"Client-{i}")
        thread.daemon = False
        client_threads.append(thread)
    
    logger.info("Starting client threads...")
    for thread in client_threads:
        thread.start()
    
    time.sleep(2)
    
    try:
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}")
    
    logger.info("Waiting for clients to finish...")
    for i, thread in enumerate(client_threads):
        thread.join(timeout=120)
        if thread.is_alive():
            logger.warning(f"Client {i} still running")
    
    logger.info("Enhanced simulation completed!")


if __name__ == "__main__":
    main()