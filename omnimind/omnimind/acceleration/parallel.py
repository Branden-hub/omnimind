import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import os

class DataParallelManager:
    """Manages data parallel training across multiple GPUs."""
    
    def __init__(self,
                 world_size: Optional[int] = None,
                 backend: str = 'nccl'):
        self.logger = logging.getLogger(__name__)
        self.world_size = world_size or torch.cuda.device_count()
        self.backend = backend
        self.initialized = False
        
        # Statistics
        self.stats = {
            'rank': -1,
            'world_size': self.world_size,
            'backend': backend
        }
    
    def initialize(self) -> None:
        """Initialize distributed training environment."""
        if self.initialized:
            return
            
        if self.world_size > 1:
            # Initialize process group
            try:
                dist.init_process_group(
                    backend=self.backend,
                    world_size=self.world_size
                )
                
                # Set device
                local_rank = dist.get_rank()
                torch.cuda.set_device(local_rank)
                
                self.stats['rank'] = local_rank
                self.initialized = True
                
                self.logger.info(
                    f"Initialized distributed training: "
                    f"rank {local_rank}/{self.world_size-1}"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to initialize distributed: {e}")
                self.world_size = 1
    
    def wrap_model(self,
                  model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training."""
        if self.world_size > 1 and self.initialized:
            # Move model to GPU
            device = torch.cuda.current_device()
            model = model.to(device)
            
            # Wrap with DDP
            model = DistributedDataParallel(
                model,
                device_ids=[device],
                output_device=device
            )
            
            self.logger.info(
                f"Model wrapped with DistributedDataParallel "
                f"on device {device}"
            )
        
        return model
    
    def prepare_dataloader(self,
                          dataloader: torch.utils.data.DataLoader,
                          shuffle: bool = True) -> torch.utils.data.DataLoader:
        """Prepare dataloader for distributed training."""
        if self.world_size > 1 and self.initialized:
            # Create distributed sampler
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataloader.dataset,
                num_replicas=self.world_size,
                rank=dist.get_rank(),
                shuffle=shuffle
            )
            
            # Create new dataloader with sampler
            return torch.utils.data.DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                sampler=sampler,
                num_workers=dataloader.num_workers,
                pin_memory=True
            )
        
        return dataloader
    
    def reduce_tensor(self,
                     tensor: torch.Tensor,
                     reduce_op: str = 'mean') -> torch.Tensor:
        """Reduce tensor across all processes."""
        if self.world_size > 1 and self.initialized:
            # All-reduce
            if reduce_op == 'mean':
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= self.world_size
            elif reduce_op == 'sum':
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            else:
                raise ValueError(f"Unknown reduce operation: {reduce_op}")
        
        return tensor
    
    def broadcast_tensor(self,
                        tensor: torch.Tensor,
                        src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank to all processes."""
        if self.world_size > 1 and self.initialized:
            dist.broadcast(tensor, src=src)
        return tensor
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.world_size > 1 and self.initialized:
            dist.barrier()
    
    def cleanup(self) -> None:
        """Clean up distributed training environment."""
        if self.initialized:
            dist.destroy_process_group()
            self.initialized = False
            self.stats['rank'] = -1
    
    def is_main_process(self) -> bool:
        """Check if current process is the main process."""
        if not self.initialized:
            return True
        return dist.get_rank() == 0
    
    def get_rank(self) -> int:
        """Get current process rank."""
        if not self.initialized:
            return 0
        return dist.get_rank()
    
    def get_world_size(self) -> int:
        """Get world size."""
        if not self.initialized:
            return 1
        return dist.get_world_size()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get distributed training statistics."""
        stats = self.stats.copy()
        
        if self.initialized:
            stats.update({
                'initialized': True,
                'current_device': torch.cuda.current_device(),
                'local_rank': self.get_rank()
            })
        else:
            stats.update({
                'initialized': False,
                'current_device': -1,
                'local_rank': -1
            })
        
        return stats
