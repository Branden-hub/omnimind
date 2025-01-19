import torch
import torch.cuda as cuda
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import psutil
import threading
import time
from pathlib import Path

class GPUManager:
    """Manages GPU resources and optimization."""
    
    def __init__(self, device_ids: Optional[List[int]] = None):
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.memory_manager = MemoryManager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize device properties
        self.device_properties = {
            device_id: torch.cuda.get_device_properties(device_id)
            for device_id in self.device_ids
        }
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitor_gpu_usage
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def optimize_memory_allocation(self,
                                 model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model memory allocation."""
        # Enable memory caching
        torch.cuda.empty_cache()
        
        # Use memory efficient optimizations
        for param in model.parameters():
            if param.requires_grad:
                param.share_memory_()
        
        return model
    
    def get_optimal_device(self,
                         required_memory: Optional[int] = None) -> torch.device:
        """Get the optimal GPU device based on current usage."""
        if not self.device_ids:
            return torch.device('cpu')
            
        # Get memory usage for each device
        memory_usage = {
            device_id: torch.cuda.memory_allocated(device_id)
            for device_id in self.device_ids
        }
        
        # Find device with most free memory
        optimal_device = min(
            memory_usage.items(),
            key=lambda x: x[1]
        )[0]
        
        # Check if device has enough memory
        if required_memory:
            free_memory = (
                self.device_properties[optimal_device].total_memory -
                memory_usage[optimal_device]
            )
            if free_memory < required_memory:
                self.logger.warning(
                    "Insufficient GPU memory. Falling back to CPU."
                )
                return torch.device('cpu')
        
        return torch.device(f'cuda:{optimal_device}')
    
    def optimize_tensor_placement(self,
                                tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Optimize tensor placement across GPUs."""
        if not self.device_ids:
            return tensors
            
        optimized_tensors = []
        current_device = 0
        
        for tensor in tensors:
            # Get tensor size
            tensor_size = tensor.element_size() * tensor.nelement()
            
            # Find optimal device
            device = self.get_optimal_device(tensor_size)
            
            # Move tensor to device
            optimized_tensors.append(tensor.to(device))
            
            # Update current device
            current_device = (current_device + 1) % len(self.device_ids)
        
        return optimized_tensors
    
    def enable_gpu_optimizations(self) -> None:
        """Enable various GPU optimizations."""
        if torch.cuda.is_available():
            # Enable cuDNN benchmarking
            torch.backends.cudnn.benchmark = True
            
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set optimal memory allocator
            torch.cuda.set_per_process_memory_fraction(0.9)
    
    def _monitor_gpu_usage(self) -> None:
        """Monitor GPU usage and log statistics."""
        while True:
            try:
                for device_id in self.device_ids:
                    # Get memory statistics
                    allocated = torch.cuda.memory_allocated(device_id)
                    cached = torch.cuda.memory_reserved(device_id)
                    
                    # Log usage
                    self.logger.info(
                        f"GPU {device_id} - Allocated: {allocated/1e9:.2f}GB, "
                        f"Cached: {cached/1e9:.2f}GB"
                    )
                    
                    # Check for memory issues
                    if allocated > 0.95 * self.device_properties[device_id].total_memory:
                        self.logger.warning(
                            f"GPU {device_id} memory usage is very high!"
                        )
                
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Error monitoring GPU usage: {e}")
    
    def get_gpu_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get current GPU statistics."""
        stats = {}
        
        for device_id in self.device_ids:
            stats[device_id] = {
                'memory_allocated': torch.cuda.memory_allocated(device_id),
                'memory_cached': torch.cuda.memory_reserved(device_id),
                'utilization': torch.cuda.utilization(device_id),
                'name': self.device_properties[device_id].name,
                'compute_capability': (
                    self.device_properties[device_id].major,
                    self.device_properties[device_id].minor
                )
            }
        
        return stats

class MemoryManager:
    """Manages GPU memory allocation and optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.allocation_history = []
    
    def allocate_tensor(self, size: Tuple[int, ...],
                       dtype: torch.dtype,
                       device: torch.device) -> torch.Tensor:
        """Allocate a tensor with memory optimization."""
        # Check cache first
        cache_key = (size, dtype, device)
        if cache_key in self.cache:
            tensor = self.cache[cache_key]
            del self.cache[cache_key]
            return tensor
        
        # Allocate new tensor
        tensor = torch.empty(size, dtype=dtype, device=device)
        
        # Record allocation
        self.allocation_history.append({
            'size': size,
            'dtype': dtype,
            'device': device,
            'timestamp': time.time()
        })
        
        return tensor
    
    def free_tensor(self, tensor: torch.Tensor) -> None:
        """Free a tensor with caching."""
        cache_key = (tensor.size(), tensor.dtype, tensor.device)
        self.cache[cache_key] = tensor
        
        # Clean cache if too large
        self._clean_cache()
    
    def _clean_cache(self, threshold: float = 0.8) -> None:
        """Clean tensor cache if memory usage is too high."""
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                memory_usage = torch.cuda.memory_allocated(device)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                
                if memory_usage > threshold * total_memory:
                    # Remove old cached tensors
                    keys_to_remove = [
                        k for k in self.cache.keys()
                        if k[2].index == device
                    ]
                    for key in keys_to_remove:
                        del self.cache[key]
                    
                    torch.cuda.empty_cache()
    
    def optimize_memory_usage(self) -> None:
        """Optimize memory usage based on allocation history."""
        if not self.allocation_history:
            return
            
        # Analyze allocation patterns
        pattern = self._analyze_allocation_pattern()
        
        # Pre-allocate frequently used tensor sizes
        for size, count in pattern.items():
            if count > 10:  # Threshold for pre-allocation
                self.allocate_tensor(
                    size[0], size[1], size[2]
                )
    
    def _analyze_allocation_pattern(self) -> Dict[Tuple, int]:
        """Analyze tensor allocation patterns."""
        pattern = {}
        
        for allocation in self.allocation_history:
            key = (
                allocation['size'],
                allocation['dtype'],
                allocation['device']
            )
            pattern[key] = pattern.get(key, 0) + 1
        
        return pattern
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            'cache_size': len(self.cache),
            'allocation_history': len(self.allocation_history)
        }
        
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                stats[f'gpu_{device}_allocated'] = torch.cuda.memory_allocated(device)
                stats[f'gpu_{device}_cached'] = torch.cuda.memory_reserved(device)
        
        return stats
