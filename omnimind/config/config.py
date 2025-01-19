import os
import secrets

# Security settings
SECRET_KEY = secrets.token_hex(32)  # Generate a secure secret key
USERNAME = "admin"  # Change this to your preferred username
PASSWORD = "your_secure_password"  # Change this to your secure password

# Server settings
HOST = "127.0.0.1"  # Only accessible from your computer
PORT = 5000

# SSL/TLS settings (for HTTPS)
SSL_ENABLED = True
SSL_CERT = "ssl/cert.pem"
SSL_KEY = "ssl/key.pem"

# Session settings
SESSION_TYPE = 'filesystem'
SESSION_PERMANENT = False
PERMANENT_SESSION_LIFETIME = 1800  # 30 minutes

# Directory settings
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
COMPONENTS_FOLDER = os.path.join(BASE_DIR, 'components')
AGENTS_FOLDER = os.path.join(BASE_DIR, 'agents')

# Create necessary directories
for folder in [UPLOAD_FOLDER, COMPONENTS_FOLDER, AGENTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
