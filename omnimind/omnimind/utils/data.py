import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    """Handles data preprocessing and validation."""
    
    def __init__(self):
        self.scalers = {}
        self.validators = {}
        self.transforms = {}
        
    def preprocess(self, data: Dict[str, torch.Tensor],
                  config: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Preprocess data according to configuration."""
        if config is None:
            config = self._default_config()
            
        processed_data = {}
        
        for key, tensor in data.items():
            # Validate data
            self._validate_data(tensor, key)
            
            # Apply transformations
            processed = self._apply_transforms(tensor, config.get('transforms', {}))
            
            # Scale data
            if config.get('scale', True):
                processed = self._scale_data(processed, key)
                
            processed_data[key] = processed
            
        return processed_data
    
    def _validate_data(self, tensor: torch.Tensor, key: str) -> None:
        """Validate data integrity and format."""
        # Check for NaN values
        if torch.isnan(tensor).any():
            raise ValueError(f"NaN values detected in {key}")
            
        # Check for infinite values
        if torch.isinf(tensor).any():
            raise ValueError(f"Infinite values detected in {key}")
            
        # Check dimensionality
        if tensor.dim() == 0:
            raise ValueError(f"Scalar tensor not supported for {key}")
            
        # Register validator for this key if not exists
        if key not in self.validators:
            self.validators[key] = {
                'shape': tensor.shape[1:],  # Expected shape excluding batch dimension
                'dtype': tensor.dtype
            }
        
    def _scale_data(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """Scale data using appropriate scaler."""
        if key not in self.scalers:
            # Initialize appropriate scaler
            if tensor.dtype in [torch.float32, torch.float64]:
                self.scalers[key] = StandardScaler()
            else:
                self.scalers[key] = MinMaxScaler()
                
            # Fit scaler
            reshaped = tensor.view(-1, tensor.shape[-1]).numpy()
            self.scalers[key].fit(reshaped)
            
        # Transform data
        reshaped = tensor.view(-1, tensor.shape[-1]).numpy()
        scaled = self.scalers[key].transform(reshaped)
        return torch.tensor(scaled).view(tensor.shape)
    
    def _apply_transforms(self, tensor: torch.Tensor,
                         transform_config: Dict[str, Any]) -> torch.Tensor:
        """Apply specified transformations to data."""
        transformed = tensor
        
        for transform_name, params in transform_config.items():
            if transform_name == 'normalize':
                transformed = self._normalize(transformed, **params)
            elif transform_name == 'standardize':
                transformed = self._standardize(transformed, **params)
            elif transform_name == 'clip':
                transformed = self._clip(transformed, **params)
                
        return transformed
    
    def _normalize(self, tensor: torch.Tensor, min_val: float = 0,
                  max_val: float = 1) -> torch.Tensor:
        """Normalize tensor to specified range."""
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
        return normalized * (max_val - min_val) + min_val
    
    def _standardize(self, tensor: torch.Tensor,
                    mean: Optional[float] = None,
                    std: Optional[float] = None) -> torch.Tensor:
        """Standardize tensor to zero mean and unit variance."""
        if mean is None:
            mean = tensor.mean()
        if std is None:
            std = tensor.std()
        return (tensor - mean) / std
    
    def _clip(self, tensor: torch.Tensor,
             min_val: Optional[float] = None,
             max_val: Optional[float] = None) -> torch.Tensor:
        """Clip tensor values to specified range."""
        return torch.clamp(tensor, min=min_val, max=max_val)
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default preprocessing configuration."""
        return {
            'scale': True,
            'transforms': {
                'standardize': {},
                'clip': {'min_val': -5, 'max_val': 5}
            }
        }

class DataValidator:
    """Validates data integrity and format."""
    
    def __init__(self):
        self.rules = {}
        
    def add_rule(self, key: str, rule: Dict[str, Any]) -> None:
        """Add validation rule for a data key."""
        self.rules[key] = rule
        
    def validate(self, data: Dict[str, torch.Tensor]) -> bool:
        """Validate data against defined rules."""
        for key, tensor in data.items():
            if key in self.rules:
                rule = self.rules[key]
                
                # Check shape
                if 'shape' in rule and tensor.shape[1:] != rule['shape']:
                    raise ValueError(
                        f"Invalid shape for {key}. Expected {rule['shape']}, "
                        f"got {tensor.shape[1:]}"
                    )
                
                # Check dtype
                if 'dtype' in rule and tensor.dtype != rule['dtype']:
                    raise ValueError(
                        f"Invalid dtype for {key}. Expected {rule['dtype']}, "
                        f"got {tensor.dtype}"
                    )
                
                # Check value range
                if 'range' in rule:
                    min_val, max_val = rule['range']
                    if tensor.min() < min_val or tensor.max() > max_val:
                        raise ValueError(
                            f"Values for {key} outside allowed range "
                            f"[{min_val}, {max_val}]"
                        )
                        
        return True

class DataTransformer:
    """Handles data transformations and augmentations."""
    
    def __init__(self):
        self.transforms = {}
        
    def add_transform(self, name: str,
                     transform_fn: callable,
                     config: Optional[Dict[str, Any]] = None) -> None:
        """Add a new transform function."""
        self.transforms[name] = {
            'function': transform_fn,
            'config': config or {}
        }
        
    def transform(self, data: torch.Tensor,
                 transform_names: List[str]) -> torch.Tensor:
        """Apply specified transforms to data."""
        transformed = data
        
        for name in transform_names:
            if name in self.transforms:
                transform = self.transforms[name]
                transformed = transform['function'](
                    transformed, **transform['config']
                )
            else:
                raise ValueError(f"Transform {name} not found")
                
        return transformed
    
    def get_available_transforms(self) -> List[str]:
        """Get list of available transforms."""
        return list(self.transforms.keys())
    
class DataAugmenter:
    """Handles data augmentation for different data types."""
    
    def __init__(self):
        self.augmenters = {}
        
    def add_augmenter(self, data_type: str,
                     augment_fn: callable,
                     config: Optional[Dict[str, Any]] = None) -> None:
        """Add an augmentation function for a data type."""
        self.augmenters[data_type] = {
            'function': augment_fn,
            'config': config or {}
        }
        
    def augment(self, data: torch.Tensor,
                data_type: str,
                num_augmentations: int = 1) -> torch.Tensor:
        """Generate augmented versions of the data."""
        if data_type not in self.augmenters:
            raise ValueError(f"No augmenter found for data type {data_type}")
            
        augmenter = self.augmenters[data_type]
        augmented = [
            augmenter['function'](data, **augmenter['config'])
            for _ in range(num_augmentations)
        ]
        
        return torch.stack(augmented)
    
class OmniMindDataset(Dataset):
    """Custom dataset for OmniMind."""
    
    def __init__(self, data: Dict[str, torch.Tensor],
                 preprocessor: Optional[DataPreprocessor] = None,
                 transformer: Optional[DataTransformer] = None,
                 augmenter: Optional[DataAugmenter] = None):
        self.data = data
        self.preprocessor = preprocessor or DataPreprocessor()
        self.transformer = transformer or DataTransformer()
        self.augmenter = augmenter or DataAugmenter()
        
        # Preprocess data
        self.processed_data = self.preprocessor.preprocess(data)
        
    def __len__(self) -> int:
        """Get dataset size."""
        return next(iter(self.processed_data.values())).shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        return {key: tensor[idx] for key, tensor in self.processed_data.items()}
