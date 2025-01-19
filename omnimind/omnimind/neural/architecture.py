import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

class DynamicArchitectureGenerator:
    """Generates neural network architectures dynamically based on requirements."""
    
    def __init__(self):
        self.layer_types = {
            'linear': nn.Linear,
            'conv1d': nn.Conv1d,
            'conv2d': nn.Conv2d,
            'lstm': nn.LSTM,
            'transformer': nn.TransformerEncoderLayer,
        }
        self.activation_types = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'gelu': nn.GELU
        }
        
    def generate(self, requirements: Dict) -> nn.Module:
        """Generate a neural network architecture based on requirements."""
        layers = []
        input_size = requirements.get('input_size', 10)
        output_size = requirements.get('output_size', 2)
        hidden_layers = requirements.get('hidden_layers', [64, 32])
        activation = requirements.get('activation', 'relu')
        dropout_rate = requirements.get('dropout', 0.1)
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(self.activation_types[activation]())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(self.activation_types[activation]())
            layers.append(nn.Dropout(dropout_rate))
            
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        
        return nn.Sequential(*layers)
    
    def generate_cnn(self, requirements: Dict) -> nn.Module:
        """Generate a CNN architecture."""
        layers = []
        in_channels = requirements.get('in_channels', 3)
        hidden_channels = requirements.get('hidden_channels', [32, 64, 128])
        kernel_size = requirements.get('kernel_size', 3)
        
        for out_channels in hidden_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(out_channels)
            ])
            in_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def generate_transformer(self, requirements: Dict) -> nn.Module:
        """Generate a transformer architecture."""
        d_model = requirements.get('d_model', 512)
        nhead = requirements.get('nhead', 8)
        num_layers = requirements.get('num_layers', 6)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=requirements.get('dim_feedforward', 2048),
            dropout=requirements.get('dropout', 0.1)
        )
        
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def generate_lstm(self, requirements: Dict) -> nn.Module:
        """Generate an LSTM architecture."""
        input_size = requirements.get('input_size', 10)
        hidden_size = requirements.get('hidden_size', 64)
        num_layers = requirements.get('num_layers', 2)
        
        return nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=requirements.get('dropout', 0.1)
        )
