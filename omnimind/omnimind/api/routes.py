from flask import Blueprint, request, jsonify, current_app
from typing import Dict, Any
import torch
import logging
import time

# Create blueprint
model_bp = Blueprint('model', __name__)
logger = logging.getLogger(__name__)

@model_bp.route('/predict', methods=['POST'])
def predict():
    """Handle model prediction requests."""
    try:
        # Get input data
        data = request.get_json()
        if not data or 'inputs' not in data:
            return jsonify({
                'error': 'No input data provided'
            }), 400
        
        # Convert to tensor
        inputs = torch.tensor(data['inputs'])
        
        # Get model server
        server = current_app.model_server
        
        # Make prediction
        start_time = time.time()
        outputs = server.predict(inputs)
        latency = time.time() - start_time
        
        # Record metrics
        current_app.monitor.record_prediction(
            inputs=inputs,
            outputs=outputs,
            latency=latency
        )
        
        return jsonify({
            'outputs': outputs.tolist(),
            'latency': latency
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@model_bp.route('/metrics', methods=['GET'])
def metrics():
    """Get model metrics."""
    try:
        metrics = current_app.monitor.get_metrics()
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@model_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time()
    })
