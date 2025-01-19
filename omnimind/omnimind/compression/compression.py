import torch
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json
from .quantization import DynamicQuantizer, StaticQuantizer
from .pruning import StructuredPruner, UnstructuredPruner
from .distillation import KnowledgeDistiller

class CompressionPipeline:
    """Pipeline for model compression using multiple techniques."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compression_config = {}
        self.stats = {}
        
        # Initialize compressors
        self.quantizer = None
        self.pruner = None
        self.distiller = None
    
    def configure_pipeline(self,
                         quantization: Optional[Dict[str, Any]] = None,
                         pruning: Optional[Dict[str, Any]] = None,
                         distillation: Optional[Dict[str, Any]] = None) -> None:
        """Configure compression pipeline."""
        # Configure quantization
        if quantization:
            method = quantization.get('method', 'dynamic')
            if method == 'dynamic':
                self.quantizer = DynamicQuantizer(
                    dtype=quantization.get('dtype')
                )
            else:
                self.quantizer = StaticQuantizer(
                    qconfig=quantization.get('qconfig')
                )
        
        # Configure pruning
        if pruning:
            method = pruning.get('method', 'unstructured')
            if method == 'structured':
                self.pruner = StructuredPruner(
                    pruning_factor=pruning.get('pruning_factor', 0.5)
                )
            else:
                self.pruner = UnstructuredPruner(
                    sparsity=pruning.get('sparsity', 0.5),
                    schedule=pruning.get('schedule')
                )
        
        # Configure distillation
        if distillation:
            self.distiller = KnowledgeDistiller(
                temperature=distillation.get('temperature', 2.0),
                alpha=distillation.get('alpha', 0.5)
            )
        
        # Store configuration
        self.compression_config = {
            'quantization': quantization,
            'pruning': pruning,
            'distillation': distillation
        }
    
    def compress_model(self,
                      model: torch.nn.Module,
                      teacher_model: Optional[torch.nn.Module] = None,
                      train_loader: Optional[torch.utils.data.DataLoader] = None,
                      val_loader: Optional[torch.utils.data.DataLoader] = None,
                      **kwargs) -> torch.nn.Module:
        """Apply compression pipeline to model."""
        compressed_model = model
        
        # Apply pruning
        if self.pruner is not None:
            self.logger.info("Applying pruning...")
            compressed_model = self.pruner.prune_model(
                compressed_model,
                **kwargs.get('pruning_args', {})
            )
            self.stats['pruning'] = self.pruner.get_statistics()
        
        # Apply quantization
        if self.quantizer is not None:
            self.logger.info("Applying quantization...")
            if isinstance(self.quantizer, StaticQuantizer) and train_loader:
                compressed_model = self.quantizer.quantize(
                    compressed_model,
                    calibration_dataloader=train_loader,
                    **kwargs.get('quantization_args', {})
                )
            else:
                compressed_model = self.quantizer.quantize(
                    compressed_model,
                    **kwargs.get('quantization_args', {})
                )
            self.stats['quantization'] = self.quantizer.get_statistics()
        
        # Apply distillation
        if self.distiller is not None and teacher_model is not None:
            self.logger.info("Applying knowledge distillation...")
            if train_loader is None:
                raise ValueError(
                    "Training dataloader required for distillation"
                )
            compressed_model = self.distiller.distill_knowledge(
                teacher_model,
                compressed_model,
                train_loader,
                val_loader,
                **kwargs.get('distillation_args', {})
            )
            self.stats['distillation'] = self.distiller.get_statistics()
        
        return compressed_model
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics."""
        return {
            'config': self.compression_config,
            'stats': self.stats
        }
    
    def save_compressed_model(self,
                            model: torch.nn.Module,
                            path: Union[str, Path]) -> None:
        """Save compressed model and configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'state_dict': model.state_dict(),
            'config': self.compression_config,
            'stats': self.stats
        }, path)
        
        # Save detailed configuration
        config_path = path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump({
                'compression_config': self.compression_config,
                'compression_stats': self.stats
            }, f, indent=2)
    
    def load_compressed_model(self,
                            model: torch.nn.Module,
                            path: Union[str, Path]) -> torch.nn.Module:
        """Load compressed model and configuration."""
        path = Path(path)
        
        # Load checkpoint
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        self.compression_config = checkpoint['config']
        self.stats = checkpoint['stats']
        
        # Reconfigure pipeline
        self.configure_pipeline(
            quantization=self.compression_config.get('quantization'),
            pruning=self.compression_config.get('pruning'),
            distillation=self.compression_config.get('distillation')
        )
        
        return model
    
    def estimate_compression_ratio(self) -> float:
        """Estimate overall compression ratio."""
        total_ratio = 1.0
        
        if 'quantization' in self.stats:
            total_ratio *= self.stats['quantization']['compression_ratio']
        
        if 'pruning' in self.stats:
            total_ratio *= (
                self.stats['pruning']['params_before'] /
                self.stats['pruning']['params_after']
            )
        
        if 'distillation' in self.stats:
            total_ratio *= self.stats['distillation']['compression_ratio']
        
        return total_ratio
