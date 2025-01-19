import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import queue
import threading
import time
from pathlib import Path
import json
import numpy as np

class ModelServer:
    """Serves models for inference with batching and queueing."""
    
    def __init__(self,
                 model: nn.Module,
                 batch_size: int = 32,
                 max_queue_size: int = 100,
                 timeout: float = 1.0):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Initialize queues
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.response_queues: Dict[str, queue.Queue] = {}
        
        # Initialize processor
        self.batch_processor = BatchProcessor(
            model=model,
            batch_size=batch_size
        )
        
        # Start processing thread
        self.running = True
        self.process_thread = threading.Thread(
            target=self._process_requests
        )
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0,
            'avg_latency': 0
        }
    
    def predict(self,
               inputs: torch.Tensor,
               request_id: Optional[str] = None) -> torch.Tensor:
        """Make prediction for single input."""
        if request_id is None:
            request_id = str(time.time())
            
        # Create response queue
        response_queue = queue.Queue()
        self.response_queues[request_id] = response_queue
        
        # Add request to queue
        try:
            self.request_queue.put(
                (request_id, inputs),
                timeout=self.timeout
            )
        except queue.Full:
            del self.response_queues[request_id]
            raise RuntimeError("Request queue is full")
        
        # Wait for response
        try:
            response = response_queue.get(timeout=self.timeout)
            del self.response_queues[request_id]
            return response
        except queue.Empty:
            del self.response_queues[request_id]
            raise RuntimeError("Response timeout")
    
    def _process_requests(self) -> None:
        """Process requests in batches."""
        batch_inputs = []
        batch_ids = []
        
        while self.running:
            try:
                # Collect batch
                while len(batch_inputs) < self.batch_size:
                    try:
                        request_id, inputs = self.request_queue.get(
                            timeout=self.timeout
                        )
                        batch_inputs.append(inputs)
                        batch_ids.append(request_id)
                    except queue.Empty:
                        break
                
                if batch_inputs:
                    # Process batch
                    start_time = time.time()
                    batch_tensor = torch.stack(batch_inputs)
                    outputs = self.batch_processor.process_batch(batch_tensor)
                    
                    # Update statistics
                    batch_time = time.time() - start_time
                    self._update_stats(len(batch_inputs), batch_time)
                    
                    # Send responses
                    for i, request_id in enumerate(batch_ids):
                        if request_id in self.response_queues:
                            self.response_queues[request_id].put(
                                outputs[i]
                            )
                    
                    # Clear batch
                    batch_inputs = []
                    batch_ids = []
                    
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                
                # Return errors to clients
                for request_id in batch_ids:
                    if request_id in self.response_queues:
                        self.response_queues[request_id].put(e)
    
    def _update_stats(self,
                     batch_size: int,
                     batch_time: float) -> None:
        """Update server statistics."""
        self.stats['total_requests'] += batch_size
        self.stats['total_batches'] += 1
        self.stats['avg_batch_size'] = (
            self.stats['total_requests'] /
            self.stats['total_batches']
        )
        self.stats['avg_latency'] = (
            (self.stats['avg_latency'] *
             (self.stats['total_batches'] - 1) +
             batch_time) /
            self.stats['total_batches']
        )
    
    def get_statistics(self) -> Dict[str, float]:
        """Get server statistics."""
        return self.stats.copy()
    
    def shutdown(self) -> None:
        """Shutdown the server."""
        self.running = False
        self.process_thread.join()

class BatchProcessor:
    """Processes batches of inputs efficiently."""
    
    def __init__(self,
                 model: nn.Module,
                 batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Move model to appropriate device
        self.device = (
            torch.device('cuda')
            if torch.cuda.is_available()
            else torch.device('cpu')
        )
        self.model = self.model.to(self.device)
        
        # Initialize inference engine
        self.inference_engine = InferenceEngine(model)
    
    def process_batch(self,
                     batch: torch.Tensor) -> torch.Tensor:
        """Process a batch of inputs."""
        try:
            # Move batch to device
            batch = batch.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.inference_engine.infer(batch)
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            raise

class InferenceEngine:
    """Optimized inference engine."""
    
    def __init__(self,
                 model: nn.Module):
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache
        self.cache = {}
        self.cache_hits = 0
        self.total_requests = 0
    
    def infer(self,
              inputs: torch.Tensor,
              use_cache: bool = True) -> torch.Tensor:
        """Run inference with caching."""
        self.total_requests += 1
        
        if use_cache:
            # Generate cache key
            cache_key = self._generate_cache_key(inputs)
            
            # Check cache
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
        
        # Run inference
        outputs = self.model(inputs)
        
        # Update cache
        if use_cache:
            self.cache[cache_key] = outputs
            self._manage_cache()
        
        return outputs
    
    def _generate_cache_key(self,
                          inputs: torch.Tensor) -> str:
        """Generate cache key for inputs."""
        # Use hash of input tensor
        return hashlib.md5(
            inputs.cpu().numpy().tobytes()
        ).hexdigest()
    
    def _manage_cache(self,
                     max_size: int = 1000) -> None:
        """Manage cache size."""
        if len(self.cache) > max_size:
            # Remove oldest entries
            num_to_remove = len(self.cache) - max_size
            for key in list(self.cache.keys())[:num_to_remove]:
                del self.cache[key]
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'hit_rate': self.cache_hits / max(1, self.total_requests),
            'total_requests': self.total_requests
        }
    
    def clear_cache(self) -> None:
        """Clear inference cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.total_requests = 0
