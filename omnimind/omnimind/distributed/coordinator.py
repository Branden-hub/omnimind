import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Union, Any
import threading
import queue
import logging
from ..utils.monitoring import PerformanceMonitor

class DistributedCoordinator:
    """Coordinates distributed training across workers."""
    
    def __init__(self, world_size: int,
                 backend: str = 'nccl',
                 timeout: float = 1800):
        self.world_size = world_size
        self.backend = backend
        self.timeout = timeout
        self.monitor = PerformanceMonitor()
        self.logger = logging.getLogger(__name__)
        
        # Communication queues
        self.command_queues = {
            i: queue.Queue() for i in range(world_size)
        }
        self.result_queues = {
            i: queue.Queue() for i in range(world_size)
        }
        
        # Worker status tracking
        self.worker_status = {
            i: {'active': True, 'last_heartbeat': 0}
            for i in range(world_size)
        }
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitor_workers
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def send_command(self, rank: int,
                    command: str,
                    payload: Optional[Dict[str, Any]] = None) -> None:
        """Send a command to a specific worker."""
        if not self.worker_status[rank]['active']:
            raise RuntimeError(f"Worker {rank} is not active")
            
        self.command_queues[rank].put({
            'command': command,
            'payload': payload or {}
        })
        
    def broadcast_command(self, command: str,
                         payload: Optional[Dict[str, Any]] = None) -> None:
        """Broadcast a command to all workers."""
        for rank in range(self.world_size):
            if self.worker_status[rank]['active']:
                self.send_command(rank, command, payload)
    
    def get_result(self, rank: int,
                  timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get result from a specific worker."""
        try:
            return self.result_queues[rank].get(
                timeout=timeout or self.timeout
            )
        except queue.Empty:
            raise TimeoutError(f"Timeout waiting for result from worker {rank}")
    
    def gather_results(self,
                      timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """Gather results from all active workers."""
        results = []
        for rank in range(self.world_size):
            if self.worker_status[rank]['active']:
                try:
                    result = self.get_result(rank, timeout)
                    results.append(result)
                except TimeoutError as e:
                    self.logger.error(f"Failed to get result from worker {rank}: {e}")
                    self.worker_status[rank]['active'] = False
        return results
    
    def synchronize_workers(self) -> None:
        """Synchronize all workers."""
        self.broadcast_command('sync')
        self.gather_results()
    
    def _monitor_workers(self) -> None:
        """Monitor worker health and handle failures."""
        while True:
            current_time = time.time()
            
            for rank, status in self.worker_status.items():
                if status['active']:
                    # Check last heartbeat
                    if current_time - status['last_heartbeat'] > self.timeout:
                        self.logger.warning(
                            f"Worker {rank} hasn't sent heartbeat for "
                            f"{self.timeout} seconds"
                        )
                        self._handle_worker_failure(rank)
            
            time.sleep(10)  # Check every 10 seconds
    
    def _handle_worker_failure(self, rank: int) -> None:
        """Handle worker failure."""
        self.logger.error(f"Worker {rank} has failed")
        self.worker_status[rank]['active'] = False
        
        # Notify other workers
        self.broadcast_command(
            'worker_failed',
            {'failed_rank': rank}
        )
        
        # Attempt recovery
        self._attempt_recovery(rank)
    
    def _attempt_recovery(self, rank: int) -> None:
        """Attempt to recover from worker failure."""
        try:
            # Initialize new process group excluding failed worker
            active_ranks = [
                r for r, status in self.worker_status.items()
                if status['active']
            ]
            
            if not active_ranks:
                raise RuntimeError("No active workers remaining")
                
            # Redistribute work
            self.broadcast_command(
                'redistribute_work',
                {'active_ranks': active_ranks}
            )
            
            self.logger.info(
                f"Successfully redistributed work after worker {rank} failure"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to recover from worker {rank} failure: {e}")
            raise
    
    def update_worker_status(self, rank: int,
                           status_update: Dict[str, Any]) -> None:
        """Update status information for a worker."""
        self.worker_status[rank].update(status_update)
        self.worker_status[rank]['last_heartbeat'] = time.time()
    
    def get_worker_status(self) -> Dict[int, Dict[str, Any]]:
        """Get current status of all workers."""
        return self.worker_status
    
    def shutdown(self) -> None:
        """Shutdown the coordinator and all workers."""
        self.broadcast_command('shutdown')
        self.monitoring_thread.join()
        
        # Clear queues
        for q in self.command_queues.values():
            while not q.empty():
                q.get()
        for q in self.result_queues.values():
            while not q.empty():
                q.get()
                
        self.logger.info("Coordinator shutdown complete")
