import torch
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import time
from pathlib import Path
import json
import threading
import queue
import psutil
import numpy as np
from datetime import datetime, timedelta

class DeploymentMonitor:
    """Monitors deployed model performance and health."""
    
    def __init__(self,
                 model_name: str,
                 metrics_window: int = 3600):
        self.model_name = model_name
        self.metrics_window = metrics_window
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            window_size=metrics_window
        )
        self.alert_manager = AlertManager(
            model_name=model_name
        )
        
        # Start monitoring thread
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def record_prediction(self,
                         inputs: torch.Tensor,
                         outputs: torch.Tensor,
                         latency: float) -> None:
        """Record a prediction event."""
        try:
            # Collect metrics
            metrics = {
                'latency': latency,
                'input_size': inputs.nelement(),
                'output_size': outputs.nelement(),
                'timestamp': time.time()
            }
            
            # Add to collector
            self.metrics_collector.add_metrics(metrics)
            
            # Check for alerts
            self.alert_manager.check_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Error recording prediction: {e}")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.metrics_collector.add_metrics(system_metrics)
                
                # Check for alerts
                self.alert_manager.check_metrics(system_metrics)
                
                # Sleep
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system metrics."""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'timestamp': time.time()
        }
        
        if torch.cuda.is_available():
            metrics['gpu_memory_percent'] = (
                torch.cuda.memory_allocated() /
                torch.cuda.max_memory_allocated() * 100
            )
        
        return metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics_collector.get_metrics()
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts."""
        return self.alert_manager.get_alerts()
    
    def shutdown(self) -> None:
        """Shutdown the monitor."""
        self.running = False
        self.monitor_thread.join()

class MetricsCollector:
    """Collects and manages metrics."""
    
    def __init__(self,
                 window_size: int = 3600):
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        self.metrics_queue = queue.Queue()
        
        # Metrics storage
        self.metrics: Dict[str, List[Tuple[float, float]]] = {}
    
    def add_metrics(self,
                   metrics: Dict[str, float]) -> None:
        """Add metrics to collection."""
        timestamp = metrics.pop('timestamp', time.time())
        
        # Add metrics to storage
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append((timestamp, value))
        
        # Clean old metrics
        self._clean_old_metrics()
    
    def _clean_old_metrics(self) -> None:
        """Remove metrics outside window."""
        cutoff_time = time.time() - self.window_size
        
        for name in self.metrics:
            self.metrics[name] = [
                (t, v) for t, v in self.metrics[name]
                if t > cutoff_time
            ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        summary = {}
        
        for name, values in self.metrics.items():
            if values:
                timestamps, metrics = zip(*values)
                summary[name] = {
                    'current': metrics[-1],
                    'mean': np.mean(metrics),
                    'std': np.std(metrics),
                    'min': np.min(metrics),
                    'max': np.max(metrics)
                }
        
        return summary

class AlertManager:
    """Manages monitoring alerts."""
    
    def __init__(self,
                 model_name: str,
                 alert_history_size: int = 1000):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.alert_history_size = alert_history_size
        
        # Alert configuration
        self.alert_rules = self._load_alert_rules()
        
        # Alert storage
        self.alerts: List[Dict[str, Any]] = []
    
    def _load_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load alert rules from configuration."""
        default_rules = {
            'high_latency': {
                'metric': 'latency',
                'threshold': 1.0,
                'condition': 'greater',
                'severity': 'warning'
            },
            'high_cpu': {
                'metric': 'cpu_percent',
                'threshold': 90,
                'condition': 'greater',
                'severity': 'warning'
            },
            'high_memory': {
                'metric': 'memory_percent',
                'threshold': 90,
                'condition': 'greater',
                'severity': 'warning'
            },
            'high_gpu_memory': {
                'metric': 'gpu_memory_percent',
                'threshold': 90,
                'condition': 'greater',
                'severity': 'warning'
            }
        }
        
        # Load custom rules if available
        rules_file = Path('config/alert_rules.json')
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                custom_rules = json.load(f)
                default_rules.update(custom_rules)
        
        return default_rules
    
    def check_metrics(self,
                     metrics: Dict[str, float]) -> None:
        """Check metrics against alert rules."""
        for rule_name, rule in self.alert_rules.items():
            metric_name = rule['metric']
            if metric_name in metrics:
                value = metrics[metric_name]
                
                if self._check_condition(
                    value,
                    rule['threshold'],
                    rule['condition']
                ):
                    self._create_alert(
                        rule_name,
                        metric_name,
                        value,
                        rule['severity']
                    )
    
    def _check_condition(self,
                        value: float,
                        threshold: float,
                        condition: str) -> bool:
        """Check if value meets alert condition."""
        if condition == 'greater':
            return value > threshold
        elif condition == 'less':
            return value < threshold
        elif condition == 'equal':
            return value == threshold
        return False
    
    def _create_alert(self,
                     rule_name: str,
                     metric_name: str,
                     value: float,
                     severity: str) -> None:
        """Create and store alert."""
        alert = {
            'timestamp': time.time(),
            'model': self.model_name,
            'rule': rule_name,
            'metric': metric_name,
            'value': value,
            'severity': severity
        }
        
        # Add to storage
        self.alerts.append(alert)
        
        # Trim history if needed
        if len(self.alerts) > self.alert_history_size:
            self.alerts = self.alerts[-self.alert_history_size:]
        
        # Log alert
        self.logger.warning(
            f"Alert: {rule_name} - {metric_name} = {value}"
        )
    
    def get_alerts(self,
                  severity: Optional[str] = None,
                  time_range: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        filtered_alerts = self.alerts
        
        if severity:
            filtered_alerts = [
                a for a in filtered_alerts
                if a['severity'] == severity
            ]
        
        if time_range:
            cutoff_time = time.time() - time_range
            filtered_alerts = [
                a for a in filtered_alerts
                if a['timestamp'] > cutoff_time
            ]
        
        return filtered_alerts
