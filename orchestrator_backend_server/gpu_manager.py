"""
GPU Manager pentru Orchestrator
Gestionează alocarea automată a GPU-urilor pentru simulări paralele
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
import multiprocessing

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU allocation for parallel simulations"""
    
    def __init__(self):
        self.available_gpus = self._detect_gpus()
        self.gpu_queue = multiprocessing.Manager().Queue()
        
        # Initialize queue with available GPUs
        for gpu_id in self.available_gpus:
            self.gpu_queue.put(gpu_id)
        
        logger.info(f"GPU Manager initialized with {len(self.available_gpus)} GPUs: {self.available_gpus}")
    
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.free', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(',')
                        gpu_id = int(parts[0].strip())
                        gpu_name = parts[1].strip()
                        memory_free = parts[2].strip()
                        logger.info(f"Detected GPU {gpu_id}: {gpu_name} ({memory_free} free)")
                        gpus.append(gpu_id)
                
                return gpus if gpus else [-1]  # Return [-1] for CPU-only if no GPUs
            else:
                logger.warning("nvidia-smi failed, falling back to CPU-only mode")
                return [-1]
                
        except FileNotFoundError:
            logger.warning("nvidia-smi not found, falling back to CPU-only mode")
            return [-1]
        except Exception as e:
            logger.error(f"Error detecting GPUs: {str(e)}")
            return [-1]
    
    def allocate_gpu(self, task_id: str, timeout: int = 300) -> int:
        """
        Allocate a GPU for a task
        Returns GPU ID or -1 for CPU
        Blocks until a GPU is available (with timeout)
        """
        try:
            gpu_id = self.gpu_queue.get(timeout=timeout)
            logger.info(f"Allocated GPU {gpu_id} for task {task_id}")
            return gpu_id
        except Exception as e:
            logger.warning(f"Could not allocate GPU for task {task_id}, using CPU: {str(e)}")
            return -1
    
    def release_gpu(self, task_id: str, gpu_id: int):
        """Release a GPU back to the pool"""
        if gpu_id != -1:  # Don't release CPU
            self.gpu_queue.put(gpu_id)
            logger.info(f"Released GPU {gpu_id} from task {task_id}")
    
    def get_gpu_memory_limit(self, gpu_id: int) -> Optional[int]:
        """
        Get recommended memory limit for GPU (in MB)
        Useful for TensorFlow memory growth
        """
        if gpu_id == -1:
            return None
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits', '-i', str(gpu_id)],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                total_memory = int(result.stdout.strip())
                # Reserve 80% of memory for safety
                return int(total_memory * 0.8)
            
        except Exception as e:
            logger.error(f"Error getting GPU memory for GPU {gpu_id}: {str(e)}")
        
        return None
    
    def get_available_count(self) -> int:
        """Get number of available GPUs in queue"""
        return self.gpu_queue.qsize()


def configure_tensorflow_gpu(gpu_id: int, memory_limit: Optional[int] = None):
    """
    Configure TensorFlow to use specific GPU with memory growth
    Must be called BEFORE any TensorFlow operations
    """
    import tensorflow as tf
    
    if gpu_id == -1:
        # CPU-only mode
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logger.info("TensorFlow configured for CPU-only mode")
        return
    
    # Set visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Configure TensorFlow GPU settings
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth to avoid conflicts
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Optionally set memory limit
            if memory_limit:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                logger.info(f"TensorFlow configured to use GPU {gpu_id} with {memory_limit}MB limit")
            else:
                logger.info(f"TensorFlow configured to use GPU {gpu_id} with memory growth")
                
        except RuntimeError as e:
            logger.error(f"Error configuring TensorFlow GPU: {str(e)}")
    else:
        logger.warning(f"GPU {gpu_id} not available to TensorFlow, falling back to CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def configure_pytorch_gpu(gpu_id: int):
    """
    Configure PyTorch to use specific GPU
    Must be called BEFORE creating any models
    """
    import torch
    
    if gpu_id == -1:
        # CPU-only mode
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logger.info("PyTorch configured for CPU-only mode")
        return 'cpu'
    
    # Set visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    if torch.cuda.is_available():
        device = f'cuda:0'  # Will be mapped to the visible GPU
        logger.info(f"PyTorch configured to use GPU {gpu_id} (cuda:0)")
        return device
    else:
        logger.warning(f"CUDA not available for PyTorch, falling back to CPU")
        return 'cpu'


# Example usage function
def run_with_gpu_allocation(gpu_manager: GPUManager, task_id: str, task_func, *args, **kwargs):
    """
    Wrapper to run a task with automatic GPU allocation/deallocation
    
    Usage:
        result = run_with_gpu_allocation(gpu_manager, task_id, my_training_function, model, data)
    """
    gpu_id = gpu_manager.allocate_gpu(task_id)
    
    try:
        # Configure framework (detect from kwargs or args)
        framework = kwargs.get('framework', 'tensorflow')
        
        if framework == 'tensorflow':
            memory_limit = gpu_manager.get_gpu_memory_limit(gpu_id)
            configure_tensorflow_gpu(gpu_id, memory_limit)
        elif framework == 'pytorch':
            device = configure_pytorch_gpu(gpu_id)
            kwargs['device'] = device
        
        # Run the task
        result = task_func(*args, **kwargs)
        return result
        
    finally:
        # Always release GPU
        gpu_manager.release_gpu(task_id, gpu_id)


if __name__ == "__main__":
    # Test GPU detection
    manager = GPUManager()
    print(f"Available GPUs: {manager.available_gpus}")
    print(f"GPUs in queue: {manager.get_available_count()}")
    
    # Test allocation/release
    task_id = "test_task_1"
    gpu = manager.allocate_gpu(task_id)
    print(f"Allocated GPU {gpu} for {task_id}")
    
    memory_limit = manager.get_gpu_memory_limit(gpu)
    print(f"Recommended memory limit: {memory_limit} MB")
    
    manager.release_gpu(task_id, gpu)
    print(f"Released GPU {gpu}")
    print(f"GPUs in queue: {manager.get_available_count()}")
