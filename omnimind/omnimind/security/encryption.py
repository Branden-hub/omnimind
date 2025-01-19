import os
import torch
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
from base64 import b64encode, b64decode
from typing import Dict, Any, Optional, Union
import logging

class BaseEncryptor:
    """Base class for encryption functionality."""
    
    def __init__(self, key: Optional[bytes] = None):
        self.key = key or self._generate_key()
        self.fernet = Fernet(self.key)
        self.logger = logging.getLogger(__name__)
    
    def _generate_key(self) -> bytes:
        """Generate a new encryption key."""
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = b64encode(kdf.derive(os.urandom(32)))
        return key
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data."""
        return self.fernet.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        return self.fernet.decrypt(encrypted_data)

class DataEncryptor(BaseEncryptor):
    """Handles encryption of sensitive data."""
    
    def __init__(self, key: Optional[bytes] = None):
        super().__init__(key)
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> Dict[str, bytes]:
        """Encrypt a PyTorch tensor."""
        # Convert tensor to bytes
        buffer = tensor.numpy().tobytes()
        
        # Encrypt data
        encrypted_data = self.encrypt(buffer)
        
        # Create metadata
        metadata = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype)
        }
        metadata_bytes = json.dumps(metadata).encode()
        encrypted_metadata = self.encrypt(metadata_bytes)
        
        return {
            'data': encrypted_data,
            'metadata': encrypted_metadata
        }
    
    def decrypt_tensor(self,
                      encrypted_dict: Dict[str, bytes]) -> torch.Tensor:
        """Decrypt a PyTorch tensor."""
        # Decrypt metadata
        metadata_bytes = self.decrypt(encrypted_dict['metadata'])
        metadata = json.loads(metadata_bytes.decode())
        
        # Decrypt data
        decrypted_buffer = self.decrypt(encrypted_dict['data'])
        
        # Reconstruct tensor
        array = np.frombuffer(decrypted_buffer,
                            dtype=np.dtype(metadata['dtype']))
        tensor = torch.from_numpy(array.reshape(metadata['shape']))
        
        return tensor
    
    def encrypt_dataset(self,
                       dataset: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Encrypt an entire dataset."""
        encrypted_dataset = {}
        for key, tensor in dataset.items():
            encrypted_dataset[key] = self.encrypt_tensor(tensor)
        return encrypted_dataset
    
    def decrypt_dataset(self,
                       encrypted_dataset: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Decrypt an entire dataset."""
        decrypted_dataset = {}
        for key, encrypted_dict in encrypted_dataset.items():
            decrypted_dataset[key] = self.decrypt_tensor(encrypted_dict)
        return decrypted_dataset

class ModelEncryptor(BaseEncryptor):
    """Handles encryption of model parameters and architecture."""
    
    def encrypt_model(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Encrypt model parameters and architecture."""
        # Get model state
        state_dict = model.state_dict()
        
        # Convert state dict to bytes
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        state_bytes = buffer.getvalue()
        
        # Encrypt state
        encrypted_state = self.encrypt(state_bytes)
        
        # Encrypt architecture
        architecture = str(model)
        encrypted_architecture = self.encrypt(architecture.encode())
        
        return {
            'state': encrypted_state,
            'architecture': encrypted_architecture
        }
    
    def decrypt_model(self, encrypted_dict: Dict[str, Any],
                     model_class: type) -> torch.nn.Module:
        """Decrypt model parameters and architecture."""
        # Decrypt architecture
        architecture = self.decrypt(
            encrypted_dict['architecture']
        ).decode()
        
        # Create new model instance
        model = model_class()
        
        # Decrypt state
        state_bytes = self.decrypt(encrypted_dict['state'])
        buffer = io.BytesIO(state_bytes)
        state_dict = torch.load(buffer)
        
        # Load state into model
        model.load_state_dict(state_dict)
        
        return model
    
    def encrypt_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt model checkpoint data."""
        encrypted_checkpoint = {}
        
        # Encrypt model state
        if 'model_state_dict' in checkpoint:
            buffer = io.BytesIO()
            torch.save(checkpoint['model_state_dict'], buffer)
            encrypted_checkpoint['model_state_dict'] = self.encrypt(
                buffer.getvalue()
            )
        
        # Encrypt optimizer state
        if 'optimizer_state_dict' in checkpoint:
            buffer = io.BytesIO()
            torch.save(checkpoint['optimizer_state_dict'], buffer)
            encrypted_checkpoint['optimizer_state_dict'] = self.encrypt(
                buffer.getvalue()
            )
        
        # Encrypt metadata
        metadata = {
            k: v for k, v in checkpoint.items()
            if k not in ['model_state_dict', 'optimizer_state_dict']
        }
        encrypted_checkpoint['metadata'] = self.encrypt(
            json.dumps(metadata).encode()
        )
        
        return encrypted_checkpoint
    
    def decrypt_checkpoint(self,
                         encrypted_checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt model checkpoint data."""
        checkpoint = {}
        
        # Decrypt model state
        if 'model_state_dict' in encrypted_checkpoint:
            buffer = io.BytesIO(
                self.decrypt(encrypted_checkpoint['model_state_dict'])
            )
            checkpoint['model_state_dict'] = torch.load(buffer)
        
        # Decrypt optimizer state
        if 'optimizer_state_dict' in encrypted_checkpoint:
            buffer = io.BytesIO(
                self.decrypt(encrypted_checkpoint['optimizer_state_dict'])
            )
            checkpoint['optimizer_state_dict'] = torch.load(buffer)
        
        # Decrypt metadata
        metadata = json.loads(
            self.decrypt(encrypted_checkpoint['metadata']).decode()
        )
        checkpoint.update(metadata)
        
        return checkpoint
