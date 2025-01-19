import torch
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json

class PruningManager:
    """Base class for model pruning."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pruning_config = {}
        self.stats = {
            'sparsity': 0.0,
            'params_before': 0,
            'params_after': 0
        }
    
    def _count_parameters(self,
                         model: torch.nn.Module) -> Tuple[int, float]:
        """Count total and zero parameters."""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        return total_params, zero_params
    
    def _update_stats(self, model: torch.nn.Module) -> None:
        """Update pruning statistics."""
        total_params, zero_params = self._count_parameters(model)
        
        self.stats.update({
            'sparsity': zero_params / total_params,
            'params_before': self.stats['params_before'],
            'params_after': total_params - zero_params
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        return self.stats.copy()
    
    def save_pruned_model(self,
                         model: torch.nn.Module,
                         path: Union[str, Path]) -> None:
        """Save pruned model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'state_dict': model.state_dict(),
            'stats': self.stats,
            'config': self.pruning_config
        }, path)
    
    def load_pruned_model(self,
                         model: torch.nn.Module,
                         path: Union[str, Path]) -> torch.nn.Module:
        """Load pruned model."""
        path = Path(path)
        
        # Load checkpoint
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        self.stats = checkpoint['stats']
        self.pruning_config = checkpoint['config']
        
        return model

class StructuredPruner(PruningManager):
    """Handles structured pruning of models."""
    
    def __init__(self,
                 pruning_factor: float = 0.5):
        super().__init__()
        self.pruning_factor = pruning_factor
    
    def prune_model(self,
                   model: torch.nn.Module,
                   importance_scores: Optional[Dict[str, torch.Tensor]] = None) -> torch.nn.Module:
        """Perform structured pruning."""
        # Store initial parameter count
        total_params, _ = self._count_parameters(model)
        self.stats['params_before'] = total_params
        
        # Prune each layer
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                # Get importance scores for layer
                if importance_scores and name in importance_scores:
                    scores = importance_scores[name]
                else:
                    scores = torch.abs(module.weight.data)
                
                # Calculate number of channels/neurons to keep
                num_total = module.weight.size(0)
                num_keep = int(num_total * (1 - self.pruning_factor))
                
                # Get indices of top channels/neurons
                _, indices = torch.topk(
                    scores.sum(dim=tuple(range(1, scores.dim()))),
                    num_keep
                )
                
                # Create pruning mask
                mask = torch.zeros_like(module.weight.data)
                mask[indices] = 1
                
                # Apply pruning
                prune.custom_from_mask(module, 'weight', mask)
        
        # Update statistics
        self._update_stats(model)
        
        # Store configuration
        self.pruning_config = {
            'type': 'structured',
            'pruning_factor': self.pruning_factor
        }
        
        return model
    
    def remove_pruning(self, model: torch.nn.Module) -> torch.nn.Module:
        """Remove pruning and make weights permanent."""
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                prune.remove(module, 'weight')
        return model

class UnstructuredPruner(PruningManager):
    """Handles unstructured pruning of models."""
    
    def __init__(self,
                 sparsity: float = 0.5,
                 schedule: Optional[str] = None):
        super().__init__()
        self.sparsity = sparsity
        self.schedule = schedule
    
    def prune_model(self,
                   model: torch.nn.Module,
                   method: str = 'l1') -> torch.nn.Module:
        """Perform unstructured pruning."""
        # Store initial parameter count
        total_params, _ = self._count_parameters(model)
        self.stats['params_before'] = total_params
        
        # Select pruning method
        if method == 'l1':
            pruning_method = prune.L1Unstructured
        elif method == 'random':
            pruning_method = prune.RandomUnstructured
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        
        # Apply pruning to each layer
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                pruning_method(
                    module,
                    name='weight',
                    amount=self.sparsity
                )
        
        # Update statistics
        self._update_stats(model)
        
        # Store configuration
        self.pruning_config = {
            'type': 'unstructured',
            'sparsity': self.sparsity,
            'method': method,
            'schedule': self.schedule
        }
        
        return model
    
    def gradual_pruning(self,
                       model: torch.nn.Module,
                       initial_sparsity: float,
                       final_sparsity: float,
                       epochs: int,
                       frequency: int = 1) -> None:
        """Apply gradual pruning during training."""
        if self.schedule != 'gradual':
            raise ValueError("Gradual pruning requires schedule='gradual'")
            
        # Calculate pruning schedule
        sparsities = torch.linspace(
            initial_sparsity,
            final_sparsity,
            epochs // frequency
        )
        
        # Store schedule in config
        self.pruning_config['gradual_schedule'] = {
            'initial_sparsity': initial_sparsity,
            'final_sparsity': final_sparsity,
            'epochs': epochs,
            'frequency': frequency
        }
        
        return sparsities
    
    def get_layer_wise_sparsity(self,
                               model: torch.nn.Module) -> Dict[str, float]:
        """Get sparsity level for each layer."""
        sparsity = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                total = module.weight.numel()
                zeros = (module.weight == 0).sum().item()
                sparsity[name] = zeros / total
        
        return sparsity
