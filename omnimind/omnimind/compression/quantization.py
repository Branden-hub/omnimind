import torch
from torch.quantization import (
    quantize_dynamic,
    quantize_static,
    prepare,
    convert,
    QConfig
)
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json

class QuantizationManager:
    """Base class for quantization management."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quantization_config = {}
        self.stats = {
            'model_size_original': 0,
            'model_size_quantized': 0,
            'compression_ratio': 1.0
        }
    
    def _get_model_size(self, model: torch.nn.Module) -> int:
        """Get model size in bytes."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size
    
    def _update_stats(self,
                     original_model: torch.nn.Module,
                     quantized_model: torch.nn.Module) -> None:
        """Update quantization statistics."""
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        
        self.stats.update({
            'model_size_original': original_size,
            'model_size_quantized': quantized_size,
            'compression_ratio': original_size / quantized_size
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get quantization statistics."""
        return self.stats.copy()
    
    def save_quantized_model(self,
                           model: torch.nn.Module,
                           path: Union[str, Path]) -> None:
        """Save quantized model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), path)
        
        # Save configuration
        config_path = path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(self.quantization_config, f, indent=2)
    
    def load_quantized_model(self,
                           model: torch.nn.Module,
                           path: Union[str, Path]) -> torch.nn.Module:
        """Load quantized model."""
        path = Path(path)
        
        # Load model
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        
        # Load configuration
        config_path = path.with_suffix('.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.quantization_config = json.load(f)
        
        return model

class DynamicQuantizer(QuantizationManager):
    """Handles dynamic quantization of models."""
    
    def __init__(self,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.dtype = dtype or torch.qint8
    
    def quantize(self,
                model: torch.nn.Module,
                qconfig_dict: Optional[Dict[str, QConfig]] = None) -> torch.nn.Module:
        """Perform dynamic quantization."""
        # Store original model size
        self.stats['model_size_original'] = self._get_model_size(model)
        
        # Configure quantization
        if qconfig_dict is None:
            qconfig_dict = {
                'object_type': [
                    (torch.nn.Linear, torch.quantization.default_dynamic_qconfig),
                    (torch.nn.LSTM, torch.quantization.default_dynamic_qconfig),
                    (torch.nn.GRU, torch.quantization.default_dynamic_qconfig)
                ]
            }
        
        # Quantize model
        quantized_model = quantize_dynamic(
            model,
            qconfig_dict=qconfig_dict,
            dtype=self.dtype
        )
        
        # Update statistics
        self._update_stats(model, quantized_model)
        
        # Store configuration
        self.quantization_config = {
            'type': 'dynamic',
            'dtype': str(self.dtype),
            'qconfig_dict': str(qconfig_dict)
        }
        
        return quantized_model

class StaticQuantizer(QuantizationManager):
    """Handles static quantization of models."""
    
    def __init__(self,
                 qconfig: Optional[QConfig] = None):
        super().__init__()
        self.qconfig = qconfig or torch.quantization.get_default_qconfig('fbgemm')
    
    def prepare_model(self,
                     model: torch.nn.Module) -> torch.nn.Module:
        """Prepare model for static quantization."""
        # Fuse modules
        model = torch.quantization.fuse_modules(
            model,
            [['conv', 'bn', 'relu']]
        )
        
        # Set qconfig
        model.qconfig = self.qconfig
        
        # Prepare model
        prepared_model = prepare(model)
        
        return prepared_model
    
    def calibrate_model(self,
                       model: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       num_batches: int = 100) -> None:
        """Calibrate model with data for static quantization."""
        model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                model(inputs)
    
    def quantize(self,
                model: torch.nn.Module,
                calibration_dataloader: Optional[torch.utils.data.DataLoader] = None,
                num_calibration_batches: int = 100) -> torch.nn.Module:
        """Perform static quantization."""
        # Store original model size
        self.stats['model_size_original'] = self._get_model_size(model)
        
        # Prepare model
        prepared_model = self.prepare_model(model)
        
        # Calibrate if dataloader provided
        if calibration_dataloader is not None:
            self.calibrate_model(
                prepared_model,
                calibration_dataloader,
                num_calibration_batches
            )
        
        # Convert model
        quantized_model = convert(prepared_model)
        
        # Update statistics
        self._update_stats(model, quantized_model)
        
        # Store configuration
        self.quantization_config = {
            'type': 'static',
            'qconfig': str(self.qconfig),
            'calibration_batches': num_calibration_batches
        }
        
        return quantized_model
    
    def get_layer_wise_quantization_stats(self,
                                        model: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
        """Get quantization statistics for each layer."""
        stats = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                stats[name] = {
                    'scale': module.weight_fake_quant.scale.item(),
                    'zero_point': module.weight_fake_quant.zero_point.item()
                }
        
        return stats
