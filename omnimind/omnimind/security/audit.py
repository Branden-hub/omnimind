import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import hashlib
import threading
import queue

class SecurityAuditor:
    """Handles security auditing and logging."""
    
    def __init__(self, log_dir: str = 'logs/security'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Initialize audit queue
        self.audit_queue = queue.Queue()
        
        # Start audit worker thread
        self.worker_thread = threading.Thread(target=self._process_audit_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def setup_logging(self) -> None:
        """Setup security logging configuration."""
        log_file = self.log_dir / 'security.log'
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_event(self, event_type: str,
                  details: Dict[str, Any],
                  severity: str = 'INFO') -> None:
        """Log a security event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'severity': severity,
            'details': details
        }
        
        # Add to audit queue
        self.audit_queue.put(event)
        
        # Log based on severity
        log_method = getattr(self.logger, severity.lower())
        log_method(f"{event_type}: {json.dumps(details)}")
    
    def _process_audit_queue(self) -> None:
        """Process events in the audit queue."""
        while True:
            try:
                event = self.audit_queue.get()
                self._write_audit_record(event)
                self.audit_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing audit event: {e}")
    
    def _write_audit_record(self, event: Dict[str, Any]) -> None:
        """Write an audit record to storage."""
        # Generate filename based on date
        date_str = datetime.now().strftime('%Y%m%d')
        audit_file = self.log_dir / f'audit_{date_str}.json'
        
        # Read existing records
        records = []
        if audit_file.exists():
            with open(audit_file, 'r') as f:
                records = json.load(f)
        
        # Add new record
        records.append(event)
        
        # Write updated records
        with open(audit_file, 'w') as f:
            json.dump(records, f, indent=2)
    
    def generate_audit_report(self,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            event_types: Optional[List[str]] = None,
                            severity: Optional[str] = None) -> Dict[str, Any]:
        """Generate an audit report for the specified criteria."""
        # Get all audit files
        audit_files = list(self.log_dir.glob('audit_*.json'))
        
        events = []
        for file in audit_files:
            with open(file, 'r') as f:
                file_events = json.load(f)
                events.extend(file_events)
        
        # Filter events
        filtered_events = self._filter_events(
            events, start_date, end_date, event_types, severity
        )
        
        # Generate statistics
        stats = self._generate_statistics(filtered_events)
        
        return {
            'period': {
                'start': start_date.isoformat() if start_date else None,
                'end': end_date.isoformat() if end_date else None
            },
            'total_events': len(filtered_events),
            'statistics': stats,
            'events': filtered_events
        }
    
    def _filter_events(self, events: List[Dict[str, Any]],
                      start_date: Optional[datetime],
                      end_date: Optional[datetime],
                      event_types: Optional[List[str]],
                      severity: Optional[str]) -> List[Dict[str, Any]]:
        """Filter events based on criteria."""
        filtered = events
        
        if start_date:
            filtered = [
                e for e in filtered
                if datetime.fromisoformat(e['timestamp']) >= start_date
            ]
            
        if end_date:
            filtered = [
                e for e in filtered
                if datetime.fromisoformat(e['timestamp']) <= end_date
            ]
            
        if event_types:
            filtered = [
                e for e in filtered
                if e['type'] in event_types
            ]
            
        if severity:
            filtered = [
                e for e in filtered
                if e['severity'] == severity
            ]
            
        return filtered
    
    def _generate_statistics(self,
                           events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics from events."""
        stats = {
            'by_type': {},
            'by_severity': {},
            'by_hour': {},
            'by_date': {}
        }
        
        for event in events:
            # Count by type
            event_type = event['type']
            stats['by_type'][event_type] = stats['by_type'].get(
                event_type, 0
            ) + 1
            
            # Count by severity
            severity = event['severity']
            stats['by_severity'][severity] = stats['by_severity'].get(
                severity, 0
            ) + 1
            
            # Count by hour and date
            timestamp = datetime.fromisoformat(event['timestamp'])
            hour = timestamp.strftime('%H:00')
            date = timestamp.strftime('%Y-%m-%d')
            
            stats['by_hour'][hour] = stats['by_hour'].get(hour, 0) + 1
            stats['by_date'][date] = stats['by_date'].get(date, 0) + 1
        
        return stats
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get current security metrics."""
        # Get today's audit file
        date_str = datetime.now().strftime('%Y%m%d')
        audit_file = self.log_dir / f'audit_{date_str}.json'
        
        if not audit_file.exists():
            return {'events_today': 0, 'severity_breakdown': {}}
        
        with open(audit_file, 'r') as f:
            events = json.load(f)
        
        # Calculate metrics
        severity_breakdown = {}
        for event in events:
            severity = event['severity']
            severity_breakdown[severity] = severity_breakdown.get(
                severity, 0
            ) + 1
        
        return {
            'events_today': len(events),
            'severity_breakdown': severity_breakdown
        }
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """Clean up old audit logs."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for file in self.log_dir.glob('audit_*.json'):
            try:
                # Extract date from filename
                date_str = file.stem.split('_')[1]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                
                if file_date < cutoff_date:
                    file.unlink()
                    self.logger.info(f"Deleted old audit file: {file}")
            except Exception as e:
                self.logger.error(f"Error cleaning up audit file {file}: {e}")
