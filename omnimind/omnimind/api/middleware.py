from functools import wraps
from flask import request, jsonify
import time
import logging

logger = logging.getLogger(__name__)

def request_logger():
    """Log request details."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Log request
            logger.info(f"{request.method} {request.path}")
            
            # Time request
            start_time = time.time()
            response = f(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log response
            logger.info(f"Response time: {duration:.3f}s")
            
            return response
        return decorated_function
    return decorator

def validate_json():
    """Validate JSON request data."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    'error': 'Content-Type must be application/json'
                }), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def rate_limit(limit: int = 100, window: int = 60):
    """Basic rate limiting."""
    def decorator(f):
        # Store request counts
        requests = {}
        
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get client IP
            client = request.remote_addr
            now = time.time()
            
            # Clean old requests
            requests[client] = [
                t for t in requests.get(client, [])
                if now - t < window
            ]
            
            # Check limit
            if len(requests[client]) >= limit:
                return jsonify({
                    'error': 'Rate limit exceeded'
                }), 429
            
            # Add request
            requests[client].append(now)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator
