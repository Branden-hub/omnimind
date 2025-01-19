import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from contextlib import contextmanager

class MixedPrecisionManager:
    """Manages mixed precision training and inference."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.enabled)
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'overflow_count': 0,
            'scale_factor': 1.0
        }
    
    @contextmanager
    def autocast(self) -> None:
        """Context manager for automatic mixed precision."""
        if self.enabled:
            with autocast():
                yield
        else:
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self,
                      optimizer: torch.optim.Optimizer,
                      loss: Optional[torch.Tensor] = None) -> None:
        """Perform optimizer step with gradient scaling."""
        if self.enabled:
            if loss is not None:
                self.scaler.scale(loss).backward()
            
            # Try to step with current scale
            try:
                self.scaler.step(optimizer)
                self.scaler.update()
                
                # Update statistics
                self.stats['scale_factor'] = self.scaler.get_scale()
                
            except RuntimeError as e:
                self.logger.warning(f"Gradient overflow detected: {e}")
                self.stats['overflow_count'] += 1
                
                # Skip step and update scaler
                self.scaler.update()
        else:
            if loss is not None:
                loss.backward()
            optimizer.step()
    
    def unscale_gradients(self,
                         optimizer: torch.optim.Optimizer) -> bool:
        """Unscale gradients for gradient clipping."""
        if self.enabled:
            return self.scaler.unscale_(optimizer)
        return True
    
    def get_dtype_for_layer(self,
                           layer_type: str) -> torch.dtype:
        """Get appropriate dtype for layer based on mixed precision rules."""
        if not self.enabled:
            return torch.float32
            
        # Use fp16 for most layers
        if layer_type in ['Linear', 'Conv2d', 'Conv3d']:
            return torch.float16
            
        # Keep some layers in fp32 for stability
        if layer_type in ['BatchNorm2d', 'LayerNorm']:
            return torch.float32
            
        return torch.float16
    
    def optimize_model_dtypes(self,
                            model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model layer dtypes for mixed precision."""
        if not self.enabled:
            return model
            
        for name, module in model.named_modules():
            layer_type = module.__class__.__name__
            dtype = self.get_dtype_for_layer(layer_type)
            
            # Convert parameters
            for param in module.parameters(recurse=False):
                param.data = param.data.to(dtype)
        
        return model
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mixed precision training statistics."""
        stats = self.stats.copy()
        
        if self.enabled:
            stats.update({
                'current_scale': self.scaler.get_scale(),
                'enabled': True
            })
        else:
            stats.update({
                'current_scale': 1.0,
                'enabled': False
            })
        
        return stats
    
    def should_skip_batch(self,
                         loss: Optional[torch.Tensor] = None) -> bool:
        """Check if batch should be skipped due to overflow."""
        if not self.enabled:
            return False
            
        if loss is not None and not torch.isfinite(loss):
            self.stats['overflow_count'] += 1
            return True
            
        return False
