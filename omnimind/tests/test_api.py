import pytest
from omnimind.api import create_app
import torch
import json

@pytest.fixture
def app():
    """Create test application."""
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'timestamp' in data

def test_predict(client):
    """Test prediction endpoint."""
    # Create test input
    inputs = torch.randn(1, 10).tolist()
    
    response = client.post(
        '/api/v1/predict',
        json={'inputs': inputs}
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'outputs' in data
    assert 'latency' in data

def test_metrics(client):
    """Test metrics endpoint."""
    response = client.get('/api/v1/metrics')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, dict)
