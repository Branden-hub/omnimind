from flask import Flask
from flask_cors import CORS
from typing import Optional, Dict, Any
import logging
from pathlib import Path
import json

from .routes import model_bp
from ..deployment.monitoring import DeploymentMonitor

class ModelServer:
    """Main API server for model serving."""
    
    def __init__(self,
                 config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize monitoring
        self.monitor = DeploymentMonitor(
            model_name=self.config.get('model_name', 'default')
        )
    
    def _load_config(self,
                    config_path: Optional[str]) -> Dict[str, Any]:
        """Load server configuration."""
        default_config = {
            'model_name': 'default',
            'host': 'localhost',
            'port': 5000,
            'debug': False,
            'cors_origins': ['*']
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                default_config.update(custom_config)
            except Exception as e:
                self.logger.warning(
                    f"Error loading config: {e}. Using defaults."
                )
        
        return default_config
    
    def create_app(self) -> Flask:
        """Create Flask application."""
        app = Flask(__name__)
        
        # Configure CORS
        CORS(app, origins=self.config['cors_origins'])
        
        # Register blueprints
        app.register_blueprint(model_bp, url_prefix='/api/v1')
        
        # Add monitoring
        app.monitor = self.monitor
        
        return app
    
    def run(self) -> None:
        """Run the server."""
        app = self.create_app()
        app.run(
            host=self.config['host'],
            port=self.config['port'],
            debug=self.config['debug']
        )

def create_app(config_path: Optional[str] = None) -> Flask:
    """Application factory function."""
    server = ModelServer(config_path)
    return server.create_app()
