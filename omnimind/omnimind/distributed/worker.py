import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Union, Any, Callable
import threading
import queue
import logging
from ..utils.monitoring import PerformanceMonitor

class DistributedWorker:
    """Worker node for distributed training."""
    
    def __init__(self, rank: int,
                 world_size: int,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: Callable,
                 device: Optional[torch.device] = None):
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or torch.device(f'cuda:{rank}')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize monitoring
        self.monitor = PerformanceMonitor()
        self.logger = logging.getLogger(__name__)
        
        # Communication queues
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Start command processing thread
        self.command_thread = threading.Thread(
            target=self._process_commands
        )
        self.command_thread.daemon = True
        self.command_thread.start()
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(
            target=self._send_heartbeat
        )
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
    
    def train_batch(self, data: torch.Tensor,
                   target: torch.Tensor) -> Dict[str, float]:
        """Train on a single batch of data."""
        self.model.train()
        
        # Move data to device
        data = data.to(self.device)
        target = target.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)
        
        # Backward pass
        loss.backward()
        
        # Synchronize gradients
        self._sync_gradients()
        
        # Update parameters
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'batch_size': data.size(0)
        }
    
    def validate_batch(self, data: torch.Tensor,
                      target: torch.Tensor) -> Dict[str, float]:
        """Validate on a single batch of data."""
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            loss = self.loss_fn(output, target)
            
            # Compute accuracy
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            
            return {
                'loss': loss.item(),
                'correct': correct,
                'batch_size': data.size(0)
            }
    
    def _sync_gradients(self) -> None:
        """Synchronize gradients across workers."""
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
    
    def _process_commands(self) -> None:
        """Process incoming commands."""
        while True:
            try:
                command = self.command_queue.get()
                
                if command['command'] == 'shutdown':
                    break
                    
                elif command['command'] == 'sync':
                    dist.barrier()
                    self.result_queue.put({'status': 'synced'})
                    
                elif command['command'] == 'train_batch':
                    result = self.train_batch(
                        command['payload']['data'],
                        command['payload']['target']
                    )
                    self.result_queue.put(result)
                    
                elif command['command'] == 'validate_batch':
                    result = self.validate_batch(
                        command['payload']['data'],
                        command['payload']['target']
                    )
                    self.result_queue.put(result)
                    
                elif command['command'] == 'worker_failed':
                    self._handle_worker_failure(
                        command['payload']['failed_rank']
                    )
                    
                elif command['command'] == 'redistribute_work':
                    self._handle_work_redistribution(
                        command['payload']['active_ranks']
                    )
                    
                else:
                    self.logger.warning(
                        f"Unknown command received: {command['command']}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error processing command: {e}")
                self.result_queue.put({'error': str(e)})
    
    def _send_heartbeat(self) -> None:
        """Send periodic heartbeat to coordinator."""
        while True:
            try:
                status = {
                    'active': True,
                    'gpu_memory': torch.cuda.memory_allocated(self.device),
                    'gpu_utilization': torch.cuda.utilization(self.device)
                }
                
                # Send status update to coordinator
                dist.send(
                    torch.tensor([1], device=self.device),
                    dst=0  # Coordinator rank
                )
                
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {e}")
                
            time.sleep(5)  # Send heartbeat every 5 seconds
    
    def _handle_worker_failure(self, failed_rank: int) -> None:
        """Handle failure of another worker."""
        self.logger.info(f"Worker {failed_rank} has failed")
        
        # Update local state
        if failed_rank < self.rank:
            self.rank -= 1
        self.world_size -= 1
        
        # Wait for work redistribution
        dist.barrier()
    
    def _handle_work_redistribution(self, active_ranks: List[int]) -> None:
        """Handle work redistribution after worker failure."""
        if self.rank not in active_ranks:
            return
            
        # Update rank in active workers
        new_rank = active_ranks.index(self.rank)
        self.rank = new_rank
        self.world_size = len(active_ranks)
        
        # Wait for all workers to update
        dist.barrier()
    
    def shutdown(self) -> None:
        """Shutdown the worker."""
        self.command_queue.put({'command': 'shutdown'})
        self.command_thread.join()
        self.heartbeat_thread.join()
        
        # Clear queues
        while not self.command_queue.empty():
            self.command_queue.get()
        while not self.result_queue.empty():
            self.result_queue.get()
            
        self.logger.info(f"Worker {self.rank} shutdown complete")
