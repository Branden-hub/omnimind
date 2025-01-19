import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
import json

class PrivacyManager:
    """Manages data privacy and protection."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.privacy_rules = {}
        self.anonymization_methods = {}
        
        # Load privacy configuration
        self._load_privacy_config()
        
        # Register default anonymization methods
        self._register_default_methods()
    
    def _load_privacy_config(self) -> None:
        """Load privacy configuration from file."""
        config_file = Path('config/privacy_config.json')
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.privacy_rules = config.get('rules', {})
    
    def _register_default_methods(self) -> None:
        """Register default anonymization methods."""
        self.register_anonymization_method(
            'hash', self._hash_anonymizer
        )
        self.register_anonymization_method(
            'mask', self._mask_anonymizer
        )
        self.register_anonymization_method(
            'noise', self._noise_anonymizer
        )
        self.register_anonymization_method(
            'categorical', self._categorical_anonymizer
        )
    
    def register_anonymization_method(self, name: str,
                                    method: callable) -> None:
        """Register a new anonymization method."""
        self.anonymization_methods[name] = method
    
    def anonymize_data(self, data: Any,
                      method: str,
                      **kwargs: Any) -> Any:
        """Anonymize data using specified method."""
        if method not in self.anonymization_methods:
            raise ValueError(f"Unknown anonymization method: {method}")
            
        return self.anonymization_methods[method](data, **kwargs)
    
    def _hash_anonymizer(self, data: Any, salt: str = '') -> str:
        """Hash-based anonymization."""
        if isinstance(data, (str, int, float)):
            return hashlib.sha256(
                f"{data}{salt}".encode()
            ).hexdigest()
        raise ValueError("Hash anonymization only supports scalar values")
    
    def _mask_anonymizer(self, data: str,
                        mask_char: str = '*',
                        show_first: int = 0,
                        show_last: int = 0) -> str:
        """Mask-based anonymization."""
        if not isinstance(data, str):
            raise ValueError("Mask anonymization only supports strings")
            
        if len(data) <= show_first + show_last:
            return data
            
        masked_part = mask_char * (len(data) - show_first - show_last)
        return data[:show_first] + masked_part + data[-show_last:]
    
    def _noise_anonymizer(self, data: Union[float, torch.Tensor],
                         scale: float = 0.1) -> Union[float, torch.Tensor]:
        """Add noise for anonymization."""
        if isinstance(data, (int, float)):
            return float(data + np.random.normal(0, scale))
        elif isinstance(data, torch.Tensor):
            return data + torch.randn_like(data) * scale
        raise ValueError(
            "Noise anonymization only supports numeric values or tensors"
        )
    
    def _categorical_anonymizer(self, data: Any,
                              mapping: Optional[Dict[Any, Any]] = None) -> Any:
        """Categorical data anonymization."""
        if mapping is None:
            # Generate random mapping
            unique_values = list(set(data))
            mapping = {
                val: f"Category_{i}"
                for i, val in enumerate(unique_values)
            }
            
        if isinstance(data, (list, tuple)):
            return [mapping.get(x, x) for x in data]
        return mapping.get(data, data)
    
    def apply_privacy_rules(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply privacy rules to data."""
        protected_data = {}
        
        for key, value in data.items():
            if key in self.privacy_rules:
                rule = self.privacy_rules[key]
                method = rule.get('method', 'mask')
                params = rule.get('params', {})
                
                protected_data[key] = self.anonymize_data(
                    value, method, **params
                )
            else:
                protected_data[key] = value
                
        return protected_data
    
    def add_privacy_rule(self, field: str,
                        method: str,
                        params: Optional[Dict[str, Any]] = None) -> None:
        """Add a new privacy rule."""
        if method not in self.anonymization_methods:
            raise ValueError(f"Unknown anonymization method: {method}")
            
        self.privacy_rules[field] = {
            'method': method,
            'params': params or {}
        }
        
        # Save updated rules
        self._save_privacy_config()
    
    def _save_privacy_config(self) -> None:
        """Save privacy configuration to file."""
        config_file = Path('config/privacy_config.json')
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump({
                'rules': self.privacy_rules
            }, f, indent=2)
    
    def check_privacy_compliance(self,
                               data: Dict[str, Any]) -> List[str]:
        """Check if data complies with privacy rules."""
        violations = []
        
        for field, rule in self.privacy_rules.items():
            if field in data:
                # Check if sensitive data is properly protected
                if not self._is_properly_protected(
                    data[field], rule['method']
                ):
                    violations.append(
                        f"Unprotected sensitive field: {field}"
                    )
        
        return violations
    
    def _is_properly_protected(self, value: Any, method: str) -> bool:
        """Check if value is properly protected."""
        if method == 'hash':
            return isinstance(value, str) and len(value) == 64
        elif method == 'mask':
            return isinstance(value, str) and '*' in value
        elif method == 'noise':
            # Cannot verify noise addition
            return True
        elif method == 'categorical':
            return isinstance(value, str) and value.startswith('Category_')
        return False
    
    def generate_privacy_report(self,
                              data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a privacy compliance report."""
        violations = self.check_privacy_compliance(data)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_fields': len(data),
            'protected_fields': len(self.privacy_rules),
            'violations': violations,
            'compliance_score': 1 - len(violations) / len(self.privacy_rules)
            if self.privacy_rules else 1.0
        }
