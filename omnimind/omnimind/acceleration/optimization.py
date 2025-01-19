import torch
import torch.cuda as cuda
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from contextlib import contextmanager

class TensorOptimizer:
    """Optimizes tensor operations for GPU acceleration."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimization_stats = {}
    
    def optimize_computation(self,
                           func: callable,
                           *args,
                           **kwargs) -> Any:
        """Optimize a computation function."""
        # Record start time
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        # Run computation in optimized context
        with torch.cuda.amp.autocast():
            result = func(*args, **kwargs)
        
        end_time.record()
        torch.cuda.synchronize()
        
        # Record statistics
        elapsed_time = start_time.elapsed_time(end_time)
        self._update_stats(func.__name__, elapsed_time)
        
        return result
    
    def optimize_memory_format(self,
                             tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory format."""
        if tensor.is_cuda:
            # Use channels last format for 4D tensors (NCHW -> NHWC)
            if tensor.dim() == 4:
                return tensor.contiguous(
                    memory_format=torch.channels_last
                )
            # Use contiguous format for other tensors
            return tensor.contiguous()
        return tensor
    
    def _update_stats(self, func_name: str, elapsed_time: float) -> None:
        """Update optimization statistics."""
        if func_name not in self.optimization_stats:
            self.optimization_stats[func_name] = {
                'count': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': float('-inf')
            }
        
        stats = self.optimization_stats[func_name]
        stats['count'] += 1
        stats['total_time'] += elapsed_time
        stats['min_time'] = min(stats['min_time'], elapsed_time)
        stats['max_time'] = max(stats['max_time'], elapsed_time)

class KernelOptimizer:
    """Optimizes CUDA kernels for better performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.kernel_cache = {}
    
    @contextmanager
    def optimized_execution(self) -> None:
        """Context manager for optimized kernel execution."""
        # Enable cuDNN autotuner
        with torch.backends.cudnn.flags(
            enabled=True,
            benchmark=True,
            deterministic=False
        ):
            yield
    
    def optimize_conv_kernel(self,
                           conv_layer: torch.nn.Conv2d) -> torch.nn.Conv2d:
        """Optimize convolution kernel."""
        # Set optimal algorithm
        if hasattr(conv_layer, '_cudnn_mode'):
            conv_layer._cudnn_mode = torch.backends.cudnn.CUDNN_FASTEST
        
        return conv_layer
    
    def optimize_matmul_kernel(self,
                             mat1: torch.Tensor,
                             mat2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimize matrix multiplication kernel."""
        # Use TF32 for better performance on Ampere GPUs
        if torch.cuda.get_device_capability()[0] >= 8:
            with torch.cuda.amp.autocast():
                mat1 = mat1.to(torch.float32)
                mat2 = mat2.to(torch.float32)
        
        return mat1, mat2

class CUDAGraphManager:
    """Manages CUDA graphs for optimized execution."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.graphs = {}
        self.warmup_steps = 3
    
    def capture_graph(self, name: str,
                     func: callable,
                     *args,
                     **kwargs) -> None:
        """Capture a CUDA graph for a function."""
        # Warmup
        for _ in range(self.warmup_steps):
            func(*args, **kwargs)
        
        # Create graph
        g = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(g):
            output = func(*args, **kwargs)
        
        self.graphs[name] = {
            'graph': g,
            'output': output
        }
    
    def run_graph(self, name: str) -> Any:
        """Run a captured CUDA graph."""
        if name not in self.graphs:
            raise ValueError(f"Graph {name} not found")
            
        graph_data = self.graphs[name]
        graph_data['graph'].replay()
        
        return graph_data['output']
    
    def clear_graphs(self) -> None:
        """Clear all captured graphs."""
        self.graphs.clear()
        torch.cuda.empty_cache()

class StreamManager:
    """Manages CUDA streams for parallel execution."""
    
    def __init__(self, num_streams: int = 4):
        self.logger = logging.getLogger(__name__)
        self.streams = [
            torch.cuda.Stream()
            for _ in range(num_streams)
        ]
        self.current_stream = 0
    
    @contextmanager
    def get_stream(self) -> torch.cuda.Stream:
        """Get next available stream."""
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % len(self.streams)
        
        with torch.cuda.stream(stream):
            yield stream
    
    def synchronize_all(self) -> None:
        """Synchronize all streams."""
        for stream in self.streams:
            stream.synchronize()
    
    def parallel_execute(self,
                        functions: List[callable],
                        *args_list,
                        **kwargs_list) -> List[Any]:
        """Execute functions in parallel using different streams."""
        results = []
        events = []
        
        for i, func in enumerate(functions):
            with self.get_stream() as stream:
                # Execute function
                result = func(
                    *args_list[i] if i < len(args_list) else (),
                    **kwargs_list[i] if i < len(kwargs_list) else {}
                )
                results.append(result)
                
                # Record event
                event = torch.cuda.Event()
                event.record(stream)
                events.append(event)
        
        # Wait for all events
        for event in events:
            event.synchronize()
        
        return results
