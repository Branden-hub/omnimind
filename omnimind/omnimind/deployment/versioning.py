import torch
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json
import shutil
from datetime import datetime

class ModelRegistry:
    """Manages model versioning and storage."""
    
    def __init__(self,
                 registry_path: Union[str, Path] = 'models'):
        self.registry_path = Path(registry_path)
        self.logger = logging.getLogger(__name__)
        
        # Create registry directory
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize version manager
        self.version_manager = VersionManager(self.registry_path)
        
        # Load registry
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry metadata."""
        registry_file = self.registry_path / 'registry.json'
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                return json.load(f)
        return {'models': {}}
    
    def _save_registry(self) -> None:
        """Save registry metadata."""
        registry_file = self.registry_path / 'registry.json'
        with open(registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self,
                      model: torch.nn.Module,
                      name: str,
                      version: str,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a model version."""
        try:
            # Create model directory
            model_dir = self.registry_path / name / version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / 'model.pt'
            torch.save(model.state_dict(), model_path)
            
            # Save metadata
            metadata = metadata or {}
            metadata.update({
                'name': name,
                'version': version,
                'created_at': datetime.now().isoformat(),
                'framework_version': torch.__version__
            })
            
            metadata_path = model_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update registry
            if name not in self.registry['models']:
                self.registry['models'][name] = {
                    'versions': [],
                    'latest_version': None
                }
            
            self.registry['models'][name]['versions'].append(version)
            self.registry['models'][name]['latest_version'] = version
            self._save_registry()
            
            return version
            
        except Exception as e:
            self.logger.error(f"Error registering model: {e}")
            raise
    
    def load_model(self,
                  name: str,
                  version: Optional[str] = None,
                  model_class: Optional[type] = None) -> torch.nn.Module:
        """Load a model version."""
        try:
            # Get version
            if version is None:
                version = self.registry['models'][name]['latest_version']
            
            # Get model path
            model_dir = self.registry_path / name / version
            model_path = model_dir / 'model.pt'
            
            # Load model
            if model_class is not None:
                model = model_class()
            else:
                model = torch.nn.Module()
            
            model.load_state_dict(torch.load(model_path))
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self,
                      name: str,
                      version: Optional[str] = None) -> Dict[str, Any]:
        """Get model information."""
        try:
            # Get version
            if version is None:
                version = self.registry['models'][name]['latest_version']
            
            # Get metadata path
            metadata_path = self.registry_path / name / version / 'metadata.json'
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            raise
    
    def delete_version(self,
                      name: str,
                      version: str) -> None:
        """Delete a model version."""
        try:
            # Remove from filesystem
            version_dir = self.registry_path / name / version
            shutil.rmtree(version_dir)
            
            # Update registry
            versions = self.registry['models'][name]['versions']
            versions.remove(version)
            
            if not versions:
                del self.registry['models'][name]
            elif self.registry['models'][name]['latest_version'] == version:
                self.registry['models'][name]['latest_version'] = versions[-1]
            
            self._save_registry()
            
        except Exception as e:
            self.logger.error(f"Error deleting version: {e}")
            raise

class VersionManager:
    """Manages model versioning."""
    
    def __init__(self,
                 registry_path: Path):
        self.registry_path = registry_path
        self.logger = logging.getLogger(__name__)
    
    def create_version(self,
                      name: str,
                      major: Optional[int] = None,
                      minor: Optional[int] = None,
                      patch: Optional[int] = None) -> str:
        """Create a new version number."""
        try:
            # Get current versions
            versions = self._get_versions(name)
            
            if not versions:
                # First version
                return '1.0.0'
            
            # Parse latest version
            latest = versions[-1]
            latest_major, latest_minor, latest_patch = map(
                int, latest.split('.')
            )
            
            # Create new version
            if major is not None:
                return f"{major}.0.0"
            elif minor is not None:
                return f"{latest_major}.{minor}.0"
            elif patch is not None:
                return f"{latest_major}.{latest_minor}.{patch}"
            else:
                return f"{latest_major}.{latest_minor}.{latest_patch + 1}"
            
        except Exception as e:
            self.logger.error(f"Error creating version: {e}")
            raise
    
    def _get_versions(self, name: str) -> List[str]:
        """Get sorted list of versions for a model."""
        model_dir = self.registry_path / name
        if not model_dir.exists():
            return []
            
        versions = []
        for version_dir in model_dir.iterdir():
            if version_dir.is_dir():
                version = version_dir.name
                if self._is_valid_version(version):
                    versions.append(version)
        
        return sorted(versions, key=self._version_key)
    
    def _is_valid_version(self, version: str) -> bool:
        """Check if version string is valid."""
        try:
            major, minor, patch = map(int, version.split('.'))
            return True
        except:
            return False
    
    def _version_key(self, version: str) -> Tuple[int, ...]:
        """Convert version string to sortable tuple."""
        return tuple(map(int, version.split('.')))
