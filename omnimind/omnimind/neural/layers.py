import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any

class LayerFactory:
    """Factory class for creating neural network layers."""
    
    def __init__(self):
        self.layer_registry = {
            'linear': self._create_linear_layer,
            'conv1d': self._create_conv1d_layer,
            'conv2d': self._create_conv2d_layer,
            'lstm': self._create_lstm_layer,
            'attention': self._create_attention_layer,
            'transformer': self._create_transformer_layer
        }
        
    def build(self, architecture: Dict[str, Any]) -> nn.Module:
        """Build a neural network from an architecture specification."""
        layers = []
        for layer_spec in architecture['layers']:
            layer_type = layer_spec['type']
            if layer_type in self.layer_registry:
                layer = self.layer_registry[layer_type](layer_spec)
                layers.append(layer)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        
        return nn.Sequential(*layers)
    
    def _create_linear_layer(self, spec: Dict) -> nn.Module:
        """Create a linear layer with optional activation and normalization."""
        layers = []
        
        # Linear transformation
        layers.append(nn.Linear(spec['in_features'], spec['out_features']))
        
        # Batch normalization
        if spec.get('batch_norm', False):
            layers.append(nn.BatchNorm1d(spec['out_features']))
            
        # Activation
        activation = spec.get('activation', 'relu')
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
            
        # Dropout
        if 'dropout' in spec:
            layers.append(nn.Dropout(spec['dropout']))
            
        return nn.Sequential(*layers)
    
    def _create_conv2d_layer(self, spec: Dict) -> nn.Module:
        """Create a 2D convolutional layer with optional pooling."""
        layers = []
        
        # Convolution
        layers.append(nn.Conv2d(
            in_channels=spec['in_channels'],
            out_channels=spec['out_channels'],
            kernel_size=spec['kernel_size'],
            stride=spec.get('stride', 1),
            padding=spec.get('padding', 0)
        ))
        
        # Batch normalization
        if spec.get('batch_norm', False):
            layers.append(nn.BatchNorm2d(spec['out_channels']))
            
        # Activation
        layers.append(nn.ReLU())
        
        # Pooling
        if spec.get('pool_size', 0) > 0:
            layers.append(nn.MaxPool2d(
                kernel_size=spec['pool_size'],
                stride=spec.get('pool_stride', None)
            ))
            
        return nn.Sequential(*layers)
    
    def _create_lstm_layer(self, spec: Dict) -> nn.Module:
        """Create an LSTM layer."""
        return nn.LSTM(
            input_size=spec['input_size'],
            hidden_size=spec['hidden_size'],
            num_layers=spec.get('num_layers', 1),
            batch_first=spec.get('batch_first', True),
            dropout=spec.get('dropout', 0)
        )
    
    def _create_attention_layer(self, spec: Dict) -> nn.Module:
        """Create a multi-head attention layer."""
        return nn.MultiheadAttention(
            embed_dim=spec['embed_dim'],
            num_heads=spec['num_heads'],
            dropout=spec.get('dropout', 0)
        )
    
    def _create_transformer_layer(self, spec: Dict) -> nn.Module:
        """Create a transformer encoder layer."""
        return nn.TransformerEncoderLayer(
            d_model=spec['d_model'],
            nhead=spec['nhead'],
            dim_feedforward=spec.get('dim_feedforward', 2048),
            dropout=spec.get('dropout', 0.1)
        )
    
    def _create_conv1d_layer(self, spec: Dict) -> nn.Module:
        """Create a 1D convolutional layer."""
        layers = []
        
        # Convolution
        layers.append(nn.Conv1d(
            in_channels=spec['in_channels'],
            out_channels=spec['out_channels'],
            kernel_size=spec['kernel_size'],
            stride=spec.get('stride', 1),
            padding=spec.get('padding', 0)
        ))
        
        # Batch normalization
        if spec.get('batch_norm', False):
            layers.append(nn.BatchNorm1d(spec['out_channels']))
            
        # Activation
        layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
