# OmniMind Agent Builder

A powerful web-based platform for building and managing AI agents with custom components, quantum processing, neural networks, and advanced cognitive systems.

## Project Structure
```
omnimind/
├── app.py              # Main Flask application
├── config.py           # Configuration settings
├── omnimind.py         # Core OmniMind functionality
├── generate_cert.py    # SSL certificate generator
├── requirements.txt    # Project dependencies
├── ssl/               # SSL certificates directory
└── templates/         # HTML templates
    ├── login.html
    ├── t1.html        # Main Dashboard
    ├── t2.html        # Quantum Interface
    ├── t3.html        # Neural Network Interface
    ├── t4.html        # Knowledge Base
    ├── agent_builder.html
    └── component_builder.html
```

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Note: This will install:
   - Flask 3.0.0
   - pyOpenSSL 23.3.0
   - cryptography 41.0.7
   - werkzeug 3.0.1
   - black 23.12.1
   - pytest 7.4.3
   - numpy 1.26.2
   - psutil 5.9.6

2. **Generate SSL Certificate**
   ```bash
   python generate_cert.py
   ```

3. **Configure Settings**
   - Open `config.py`
   - Change the default username and password
   - Save the file

4. **Run the Application**
   ```bash
   python app.py
   ```

## Accessing the Application

1. Open your web browser
2. Go to `https://127.0.0.1:5000`
3. You'll see a security warning (normal for self-signed certificates)
4. Click "Advanced" then "Proceed anyway"
5. Log in with your credentials

## Features

- **Agent Builder**: Drag-and-drop interface for creating AI agents
- **Component Builder**: Create and manage custom components
- **Reference System**: Comprehensive coding terms and definitions
- **Testing Suite**: Unit, integration, and performance testing
- **Documentation Generator**: Automatic documentation for components
- **Visualization**: Component relationship mapping
- **Security**: Local-only access, password protection, HTTPS

## Security Features

- HTTPS encryption (self-signed certificate)
- Password protection
- Local-only access (127.0.0.1)
- Session management
- Secure password handling
- Activity logging

## Important Notes

1. **First Time Setup**: Make sure to complete all installation steps before running
2. **Security**: Change the default password in config.py
3. **SSL Certificate**: The self-signed certificate warning is normal for local development
4. **Backup**: Regularly backup your custom components and configurations

## Future Enhancements (Planned)

- Two-factor authentication
- Automated backups
- Rate limiting
- Advanced component testing
- More visualization options
- Additional reference categories

## Support

For issues or questions, please check the documentation or create an issue in the repository.

## License

Private use only. All rights reserved.
