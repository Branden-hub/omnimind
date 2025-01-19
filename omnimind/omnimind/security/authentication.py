import jwt
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import bcrypt
import logging
from pathlib import Path
import json

class AuthManager:
    """Manages authentication and authorization."""
    
    def __init__(self, secret_key: Optional[str] = None,
                 token_expiry: int = 3600):
        self.secret_key = secret_key or self._generate_secret_key()
        self.token_expiry = token_expiry
        self.users = {}
        self.sessions = {}
        self.logger = logging.getLogger(__name__)
        
        # Load existing users if available
        self._load_users()
    
    def _generate_secret_key(self) -> str:
        """Generate a new secret key."""
        return str(uuid.uuid4())
    
    def _load_users(self) -> None:
        """Load user data from storage."""
        user_file = Path('config/users.json')
        if user_file.exists():
            with open(user_file, 'r') as f:
                self.users = json.load(f)
    
    def _save_users(self) -> None:
        """Save user data to storage."""
        user_file = Path('config/users.json')
        user_file.parent.mkdir(parents=True, exist_ok=True)
        with open(user_file, 'w') as f:
            json.dump(self.users, f)
    
    def register_user(self, username: str,
                     password: str,
                     roles: Optional[List[str]] = None) -> bool:
        """Register a new user."""
        if username in self.users:
            return False
        
        # Hash password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode(), salt)
        
        # Store user
        self.users[username] = {
            'password': hashed.decode(),
            'roles': roles or ['user'],
            'created_at': datetime.now().isoformat()
        }
        
        self._save_users()
        return True
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return token."""
        if username not in self.users:
            return None
        
        # Verify password
        user = self.users[username]
        if not bcrypt.checkpw(password.encode(),
                            user['password'].encode()):
            return None
        
        # Generate token
        token = self._generate_token(username, user['roles'])
        
        # Store session
        self.sessions[token] = {
            'username': username,
            'created_at': time.time()
        }
        
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate token and return payload."""
        try:
            # Check if session exists
            if token not in self.sessions:
                return None
            
            session = self.sessions[token]
            
            # Check token expiry
            if time.time() - session['created_at'] > self.token_expiry:
                del self.sessions[token]
                return None
            
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
            
        except jwt.InvalidTokenError:
            return None
    
    def _generate_token(self, username: str,
                       roles: List[str]) -> str:
        """Generate JWT token."""
        payload = {
            'username': username,
            'roles': roles,
            'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def revoke_token(self, token: str) -> None:
        """Revoke a token."""
        if token in self.sessions:
            del self.sessions[token]
    
    def has_role(self, token: str, role: str) -> bool:
        """Check if token has specific role."""
        payload = self.validate_token(token)
        if not payload:
            return False
        return role in payload['roles']

class TokenManager:
    """Manages API tokens and access control."""
    
    def __init__(self):
        self.tokens = {}
        self.permissions = {}
        self.logger = logging.getLogger(__name__)
        
        # Load existing tokens if available
        self._load_tokens()
    
    def _load_tokens(self) -> None:
        """Load token data from storage."""
        token_file = Path('config/tokens.json')
        if token_file.exists():
            with open(token_file, 'r') as f:
                data = json.load(f)
                self.tokens = data.get('tokens', {})
                self.permissions = data.get('permissions', {})
    
    def _save_tokens(self) -> None:
        """Save token data to storage."""
        token_file = Path('config/tokens.json')
        token_file.parent.mkdir(parents=True, exist_ok=True)
        with open(token_file, 'w') as f:
            json.dump({
                'tokens': self.tokens,
                'permissions': self.permissions
            }, f)
    
    def create_token(self, name: str,
                    permissions: List[str],
                    expiry: Optional[int] = None) -> str:
        """Create a new API token."""
        token = str(uuid.uuid4())
        
        self.tokens[token] = {
            'name': name,
            'created_at': time.time(),
            'expiry': expiry,
            'permissions': permissions
        }
        
        self._save_tokens()
        return token
    
    def validate_token(self, token: str) -> bool:
        """Validate API token."""
        if token not in self.tokens:
            return False
            
        token_data = self.tokens[token]
        
        # Check expiry
        if token_data['expiry']:
            if time.time() - token_data['created_at'] > token_data['expiry']:
                del self.tokens[token]
                self._save_tokens()
                return False
        
        return True
    
    def check_permission(self, token: str,
                        permission: str) -> bool:
        """Check if token has specific permission."""
        if not self.validate_token(token):
            return False
            
        return permission in self.tokens[token]['permissions']
    
    def revoke_token(self, token: str) -> None:
        """Revoke an API token."""
        if token in self.tokens:
            del self.tokens[token]
            self._save_tokens()
    
    def list_tokens(self) -> List[Dict[str, Any]]:
        """List all active tokens."""
        return [
            {
                'token': token,
                'name': data['name'],
                'created_at': data['created_at'],
                'expiry': data['expiry'],
                'permissions': data['permissions']
            }
            for token, data in self.tokens.items()
        ]
    
    def add_permission(self, permission: str,
                      description: str) -> None:
        """Add a new permission type."""
        self.permissions[permission] = {
            'description': description,
            'created_at': time.time()
        }
        self._save_tokens()
    
    def remove_permission(self, permission: str) -> None:
        """Remove a permission type."""
        if permission in self.permissions:
            del self.permissions[permission]
            
            # Remove permission from all tokens
            for token_data in self.tokens.values():
                if permission in token_data['permissions']:
                    token_data['permissions'].remove(permission)
            
            self._save_tokens()
    
    def list_permissions(self) -> Dict[str, str]:
        """List all available permissions."""
        return {
            perm: data['description']
            for perm, data in self.permissions.items()
        }
