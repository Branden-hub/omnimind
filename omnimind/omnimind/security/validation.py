import re
import hashlib
import json
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path

class SecurityValidator:
    """Validates security-related aspects of the system."""
    
    def __init__(self):
        self.rules = {}
        self.violations = []
        self.logger = logging.getLogger(__name__)
        
        # Load security rules
        self._load_rules()
    
    def _load_rules(self) -> None:
        """Load security rules from configuration."""
        rules_file = Path('config/security_rules.json')
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                self.rules = json.load(f)
    
    def validate_data(self, data: Any) -> List[str]:
        """Validate data against security rules."""
        self.violations = []
        
        if isinstance(data, dict):
            self._validate_dict(data)
        elif isinstance(data, (list, tuple)):
            self._validate_sequence(data)
        else:
            self._validate_value(data)
        
        return self.violations
    
    def _validate_dict(self, data: Dict[str, Any]) -> None:
        """Validate dictionary data."""
        # Check for sensitive keys
        sensitive_patterns = self.rules.get('sensitive_keys', [])
        for key in data.keys():
            for pattern in sensitive_patterns:
                if re.match(pattern, key, re.IGNORECASE):
                    self.violations.append(
                        f"Sensitive key found: {key}"
                    )
        
        # Validate values
        for value in data.values():
            if isinstance(value, dict):
                self._validate_dict(value)
            elif isinstance(value, (list, tuple)):
                self._validate_sequence(value)
            else:
                self._validate_value(value)
    
    def _validate_sequence(self, data: Union[List, tuple]) -> None:
        """Validate sequence data."""
        for item in data:
            if isinstance(item, dict):
                self._validate_dict(item)
            elif isinstance(item, (list, tuple)):
                self._validate_sequence(item)
            else:
                self._validate_value(item)
    
    def _validate_value(self, value: Any) -> None:
        """Validate individual value."""
        if isinstance(value, str):
            self._validate_string(value)
    
    def _validate_string(self, value: str) -> None:
        """Validate string value."""
        # Check for potential SQL injection
        sql_patterns = self.rules.get('sql_injection_patterns', [])
        for pattern in sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                self.violations.append(
                    f"Potential SQL injection detected: {value}"
                )
        
        # Check for potential XSS
        xss_patterns = self.rules.get('xss_patterns', [])
        for pattern in xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                self.violations.append(
                    f"Potential XSS detected: {value}"
                )
        
        # Check for sensitive data patterns
        sensitive_patterns = self.rules.get('sensitive_data_patterns', [])
        for pattern in sensitive_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                self.violations.append(
                    f"Sensitive data pattern detected: {value}"
                )
    
    def validate_model(self, model: Any) -> List[str]:
        """Validate model security."""
        self.violations = []
        
        # Check model serialization
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                self._validate_model_architecture(model)
        except ImportError:
            pass
        
        return self.violations
    
    def _validate_model_architecture(self, model: 'torch.nn.Module') -> None:
        """Validate model architecture for security concerns."""
        # Check for known vulnerable layers
        vulnerable_layers = self.rules.get('vulnerable_layers', [])
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type in vulnerable_layers:
                self.violations.append(
                    f"Potentially vulnerable layer found: {name} ({module_type})"
                )
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration for security issues."""
        self.violations = []
        
        # Check for exposed credentials
        credential_patterns = self.rules.get('credential_patterns', [])
        self._check_credentials(config, credential_patterns)
        
        # Check for insecure settings
        insecure_settings = self.rules.get('insecure_settings', {})
        self._check_settings(config, insecure_settings)
        
        return self.violations
    
    def _check_credentials(self, config: Dict[str, Any],
                         patterns: List[str]) -> None:
        """Check for exposed credentials in configuration."""
        for key, value in config.items():
            if isinstance(value, dict):
                self._check_credentials(value, patterns)
            elif isinstance(value, str):
                for pattern in patterns:
                    if re.match(pattern, key, re.IGNORECASE):
                        self.violations.append(
                            f"Exposed credential in config: {key}"
                        )
    
    def _check_settings(self, config: Dict[str, Any],
                       insecure_settings: Dict[str, Any]) -> None:
        """Check for insecure settings in configuration."""
        for key, value in config.items():
            if isinstance(value, dict):
                self._check_settings(value, insecure_settings)
            else:
                if key in insecure_settings:
                    secure_value = insecure_settings[key]
                    if value != secure_value:
                        self.violations.append(
                            f"Insecure setting found: {key}={value}"
                        )
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate a comprehensive security report."""
        return {
            'total_violations': len(self.violations),
            'violations': self.violations,
            'rules_checked': len(self.rules),
            'timestamp': datetime.now().isoformat()
        }
    
    def add_rule(self, rule_type: str, rule_data: Any) -> None:
        """Add a new security rule."""
        if rule_type not in self.rules:
            self.rules[rule_type] = []
        
        if isinstance(rule_data, list):
            self.rules[rule_type].extend(rule_data)
        else:
            self.rules[rule_type].append(rule_data)
        
        # Save updated rules
        rules_file = Path('config/security_rules.json')
        rules_file.parent.mkdir(parents=True, exist_ok=True)
        with open(rules_file, 'w') as f:
            json.dump(self.rules, f, indent=2)
