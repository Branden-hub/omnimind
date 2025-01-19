import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import Dict, List, Optional, Union, Any, Tuple
from ..utils.checkpointing import ModelCheckpointer
from ..utils.monitoring import PerformanceMonitor
import logging

class DistributedTrainer:
    """Handles distributed training across multiple devices/nodes."""
    
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 strategy: str = 'data_parallel',
                 world_size: int = 1,
                 rank: int = 0,
                 backend: str = 'nccl',
                 master_addr: str = 'localhost',
                 master_port: str = '12355'):
        self.model = model
        self.optimizer = optimizer
        self.strategy = strategy
        self.world_size = world_size
        self.rank = rank
        self.backend = backend
        
        # Initialize distributed environment
        self._init_distributed(master_addr, master_port)
        
        # Setup model for distributed training
        self.model = self._setup_model()
        
        # Initialize monitoring
        self.monitor = PerformanceMonitor()
        self.logger = logging.getLogger(__name__)
        
        # Initialize checkpointing
        self.checkpointer = ModelCheckpointer()
        
    def _init_distributed(self, master_addr: str, master_port: str) -> None:
        """Initialize distributed training environment."""
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        dist.init_process_group(
            backend=self.backend,
            init_method=f'tcp://{master_addr}:{master_port}',
            world_size=self.world_size,
            rank=self.rank
        )
        
    def _setup_model(self) -> torch.nn.Module:
        """Setup model for distributed training."""
        if self.strategy == 'data_parallel':
            return DistributedDataParallel(
                self.model.cuda(self.rank),
                device_ids=[self.rank]
            )
        elif self.strategy == 'model_parallel':
            return self._setup_model_parallel()
        elif self.strategy == 'pipeline_parallel':
            return self._setup_pipeline_parallel()
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
    
    def _setup_model_parallel(self) -> torch.nn.Module:
        """Setup model for model parallelism."""
        # Implement model parallel setup
        raise NotImplementedError("Model parallelism not yet implemented")
    
    def _setup_pipeline_parallel(self) -> torch.nn.Module:
        """Setup model for pipeline parallelism."""
        # Implement pipeline parallel setup
        raise NotImplementedError("Pipeline parallelism not yet implemented")
    
    def train(self, train_loader: torch.utils.data.DataLoader,
              val_loader: Optional[torch.utils.data.DataLoader] = None,
              epochs: int = 10,
              checkpoint_freq: int = 1) -> Dict[str, List[float]]:
        """Train the model in a distributed setting."""
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                
            # Checkpointing
            if self.rank == 0 and epoch % checkpoint_freq == 0:
                self._save_checkpoint(epoch, {
                    'train_loss': train_loss,
                    'val_loss': val_loss if val_loader else None
                })
            
            # Synchronize processes
            dist.barrier()
        
        return history
    
    def _train_epoch(self,
                    train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(self.rank), target.cuda(self.rank)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self._compute_loss(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient synchronization
            self._sync_gradients()
            
            # Update parameters
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Rank {self.rank}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        return total_loss / num_batches
    
    def _validate(self,
                 val_loader: torch.utils.data.DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.cuda(self.rank), target.cuda(self.rank)
                output = self.model(data)
                loss = self._compute_loss(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _compute_loss(self, output: torch.Tensor,
                     target: torch.Tensor) -> torch.Tensor:
        """Compute the loss function."""
        return torch.nn.functional.cross_entropy(output, target)
    
    def _sync_gradients(self) -> None:
        """Synchronize gradients across processes."""
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
    
    def _save_checkpoint(self, epoch: int,
                        metrics: Dict[str, float]) -> None:
        """Save a checkpoint of the model."""
        if self.rank == 0:  # Only save on master process
            self.checkpointer.save_checkpoint(
                model=self.model.module,  # Get the underlying model
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=metrics
            )
    
    def load_checkpoint(self, checkpoint_id: str) -> None:
        """Load a checkpoint."""
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        self.checkpointer.load_checkpoint(
            checkpoint_id=checkpoint_id,
            model=self.model.module,
            optimizer=self.optimizer,
            device=torch.device(f'cuda:{self.rank}')
        )
    
    def cleanup(self) -> None:
        """Cleanup distributed training resources."""
        dist.destroy_process_group()
