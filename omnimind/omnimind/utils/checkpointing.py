import os
import json
import torch
import shutil
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import hashlib
from pathlib import Path

class ModelCheckpointer:
    """Handles model checkpointing and versioning."""
    
    def __init__(self, base_dir: str = 'model_checkpoints',
                 max_checkpoints: int = 5):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_history = []
        self.metadata_file = self.base_dir / 'checkpoint_metadata.json'
        self._load_metadata()
        
    def _load_metadata(self) -> None:
        """Load checkpoint metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                self.checkpoint_history = metadata.get('history', [])
        
    def _save_metadata(self) -> None:
        """Save checkpoint metadata to disk."""
        metadata = {
            'history': self.checkpoint_history,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_checkpoint(self, model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       epoch: Optional[int] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       custom_data: Optional[Dict[str, Any]] = None) -> str:
        """Save a model checkpoint."""
        # Generate checkpoint ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_hash = self._compute_model_hash(model)
        checkpoint_id = f"{timestamp}_{model_hash[:8]}"
        
        # Create checkpoint directory
        checkpoint_dir = self.base_dir / checkpoint_id
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model state
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics or {},
            'custom_data': custom_data or {},
            'timestamp': timestamp,
            'model_hash': model_hash
        }
        
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_dir / 'checkpoint.pt')
        
        # Save model architecture
        self._save_model_architecture(model, checkpoint_dir)
        
        # Update checkpoint history
        self.checkpoint_history.append({
            'id': checkpoint_id,
            'timestamp': timestamp,
            'metrics': metrics or {},
            'epoch': epoch
        })
        
        # Maintain max checkpoints limit
        self._cleanup_old_checkpoints()
        
        # Save metadata
        self._save_metadata()
        
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Load a model checkpoint."""
        checkpoint_dir = self.base_dir / checkpoint_id
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        # Load checkpoint data
        checkpoint_data = torch.load(
            checkpoint_dir / 'checkpoint.pt',
            map_location=device or torch.device('cpu')
        )
        
        # Verify model hash
        current_hash = self._compute_model_hash(model)
        if current_hash != checkpoint_data['model_hash']:
            raise ValueError(
                "Model architecture mismatch. The checkpoint was created with a "
                "different model architecture."
            )
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        return checkpoint_data
    
    def get_best_checkpoint(self,
                          metric_name: str,
                          mode: str = 'max') -> Optional[str]:
        """Get the checkpoint ID with the best metric value."""
        if not self.checkpoint_history:
            return None
            
        sorted_checkpoints = sorted(
            self.checkpoint_history,
            key=lambda x: x.get('metrics', {}).get(metric_name, float('-inf')),
            reverse=(mode == 'max')
        )
        
        return sorted_checkpoints[0]['id'] if sorted_checkpoints else None
    
    def _compute_model_hash(self, model: torch.nn.Module) -> str:
        """Compute a hash of the model architecture."""
        model_str = str(model)
        return hashlib.sha256(model_str.encode()).hexdigest()
    
    def _save_model_architecture(self, model: torch.nn.Module,
                               checkpoint_dir: Path) -> None:
        """Save model architecture information."""
        architecture_info = {
            'model_str': str(model),
            'num_parameters': sum(
                p.numel() for p in model.parameters()
            ),
            'trainable_parameters': sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        }
        
        with open(checkpoint_dir / 'architecture.json', 'w') as f:
            json.dump(architecture_info, f, indent=2)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max_checkpoints limit."""
        if len(self.checkpoint_history) > self.max_checkpoints:
            # Sort by timestamp
            sorted_checkpoints = sorted(
                self.checkpoint_history,
                key=lambda x: x['timestamp']
            )
            
            # Remove oldest checkpoints
            checkpoints_to_remove = sorted_checkpoints[
                :len(self.checkpoint_history) - self.max_checkpoints
            ]
            
            for checkpoint in checkpoints_to_remove:
                checkpoint_dir = self.base_dir / checkpoint['id']
                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)
                self.checkpoint_history.remove(checkpoint)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return self.checkpoint_history
    
    def remove_checkpoint(self, checkpoint_id: str) -> None:
        """Remove a specific checkpoint."""
        checkpoint_dir = self.base_dir / checkpoint_id
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            
        self.checkpoint_history = [
            cp for cp in self.checkpoint_history if cp['id'] != checkpoint_id
        ]
        self._save_metadata()
        
class VersionManager:
    """Manages model versions and their metadata."""
    
    def __init__(self, base_dir: str = 'model_versions'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.base_dir / 'versions.json'
        self.versions = self._load_versions()
        
    def _load_versions(self) -> Dict[str, Any]:
        """Load versions metadata from disk."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return {'versions': [], 'latest': None}
    
    def _save_versions(self) -> None:
        """Save versions metadata to disk."""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def create_version(self, checkpoint_id: str,
                      version_name: str,
                      description: str = '',
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new version from a checkpoint."""
        version_id = f"{version_name}_{datetime.now().strftime('%Y%m%d')}"
        
        version_info = {
            'id': version_id,
            'name': version_name,
            'checkpoint_id': checkpoint_id,
            'description': description,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        self.versions['versions'].append(version_info)
        self.versions['latest'] = version_id
        self._save_versions()
        
        return version_id
    
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific version."""
        for version in self.versions['versions']:
            if version['id'] == version_id:
                return version
        return None
    
    def get_latest_version(self) -> Optional[Dict[str, Any]]:
        """Get the latest version information."""
        if self.versions['latest']:
            return self.get_version(self.versions['latest'])
        return None
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all available versions."""
        return self.versions['versions']
