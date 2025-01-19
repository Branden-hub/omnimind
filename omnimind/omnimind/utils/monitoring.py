import time
import psutil
import logging
import torch
from typing import Dict, List, Optional, Union, Any
from prometheus_client import Counter, Gauge, Histogram
import wandb

class PerformanceMonitor:
    """Monitors system and model performance metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics = {}
        self.start_time = time.time()
        
        # Initialize prometheus metrics
        self.memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        self.gpu_usage = Gauge('gpu_usage_percent', 'GPU usage percentage')
        self.inference_time = Histogram(
            'inference_time_seconds',
            'Time taken for inference',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
        self.error_counter = Counter(
            'error_count',
            'Number of errors encountered'
        )
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize W&B if configured
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'omnimind'),
                config=self.config
            )
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/performance.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self) -> None:
        """Start monitoring system metrics."""
        self.start_time = time.time()
        self._update_system_metrics()
    
    def _update_system_metrics(self) -> None:
        """Update system performance metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        self.metrics['cpu_usage'] = cpu_percent
        self.cpu_usage.set(cpu_percent)
        
        # Memory metrics
        memory = psutil.Process().memory_info()
        self.metrics['memory_usage'] = memory.rss
        self.memory_usage.set(memory.rss)
        
        # GPU metrics if available
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.metrics['gpu_usage'] = gpu_usage * 100
            self.gpu_usage.set(gpu_usage * 100)
    
    def log_inference(self, duration: float) -> None:
        """Log inference time."""
        self.inference_time.observe(duration)
        self.metrics['last_inference_time'] = duration
        
        if self.config.get('use_wandb', False):
            wandb.log({'inference_time': duration})
    
    def log_error(self, error: Exception) -> None:
        """Log an error."""
        self.error_counter.inc()
        self.logger.error(f"Error encountered: {str(error)}")
        
        if self.config.get('use_wandb', False):
            wandb.log({'error_count': self.error_counter._value.get()})
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        self._update_system_metrics()
        return self.metrics
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log custom metrics."""
        self.metrics.update(metrics)
        
        for name, value in metrics.items():
            self.logger.info(f"{name}: {value}")
            
            if self.config.get('use_wandb', False):
                wandb.log({name: value})
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        uptime = time.time() - self.start_time
        self._update_system_metrics()
        
        return {
            'uptime': uptime,
            'total_errors': self.error_counter._value.get(),
            'current_metrics': self.metrics
        }

class ErrorHandler:
    """Handles error logging and recovery strategies."""
    
    def __init__(self):
        self.error_history = []
        self.recovery_strategies = {}
        self.logger = logging.getLogger(__name__)
        
    def register_recovery_strategy(self, error_type: type,
                                 strategy: callable) -> None:
        """Register a recovery strategy for an error type."""
        self.recovery_strategies[error_type] = strategy
        
    def handle_error(self, error: Exception,
                    context: Optional[Dict[str, Any]] = None) -> None:
        """Handle an error with appropriate recovery strategy."""
        # Log error
        self.logger.error(f"Error encountered: {str(error)}", exc_info=True)
        
        # Record error
        self.error_history.append({
            'error': error,
            'context': context,
            'timestamp': time.time()
        })
        
        # Attempt recovery
        error_type = type(error)
        if error_type in self.recovery_strategies:
            try:
                self.recovery_strategies[error_type](error, context)
                self.logger.info(f"Recovery strategy executed for {error_type}")
            except Exception as recovery_error:
                self.logger.error(
                    f"Recovery strategy failed: {str(recovery_error)}",
                    exc_info=True
                )
        else:
            self.logger.warning(f"No recovery strategy found for {error_type}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        return {
            'total_errors': len(self.error_history),
            'error_types': self._count_error_types(),
            'recent_errors': self.error_history[-10:]  # Last 10 errors
        }
    
    def _count_error_types(self) -> Dict[str, int]:
        """Count occurrences of each error type."""
        counts = {}
        for error_entry in self.error_history:
            error_type = type(error_entry['error']).__name__
            counts[error_type] = counts.get(error_type, 0) + 1
        return counts

class MetricsCollector:
    """Collects and aggregates various system metrics."""
    
    def __init__(self):
        self.metrics_history = []
        self.current_metrics = {}
        self.aggregations = {}
        
    def record_metric(self, name: str, value: float,
                     timestamp: Optional[float] = None) -> None:
        """Record a metric value."""
        if timestamp is None:
            timestamp = time.time()
            
        metric_entry = {
            'name': name,
            'value': value,
            'timestamp': timestamp
        }
        
        self.metrics_history.append(metric_entry)
        self.current_metrics[name] = value
        
        # Update aggregations
        if name not in self.aggregations:
            self.aggregations[name] = {
                'count': 0,
                'sum': 0,
                'min': float('inf'),
                'max': float('-inf')
            }
            
        agg = self.aggregations[name]
        agg['count'] += 1
        agg['sum'] += value
        agg['min'] = min(agg['min'], value)
        agg['max'] = max(agg['max'], value)
    
    def get_metric(self, name: str) -> Optional[float]:
        """Get current value of a metric."""
        return self.current_metrics.get(name)
    
    def get_metric_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if name not in self.aggregations:
            return {}
            
        agg = self.aggregations[name]
        return {
            'count': agg['count'],
            'mean': agg['sum'] / agg['count'],
            'min': agg['min'],
            'max': agg['max']
        }
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics."""
        return {
            name: self.get_metric_statistics(name)
            for name in self.aggregations
        }
