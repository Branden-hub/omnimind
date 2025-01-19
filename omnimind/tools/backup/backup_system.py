import os
import shutil
import datetime
import json
from pathlib import Path

class BackupSystem:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.backup_dir = os.path.join(self.base_dir, 'backups')
        self.backup_config = os.path.join(self.base_dir, 'backup_config.json')
        self.initialize_backup_system()

    def initialize_backup_system(self):
        """Create backup directory and config if they don't exist"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
        
        if not os.path.exists(self.backup_config):
            default_config = {
                'last_backup': None,
                'backup_frequency_hours': 24,
                'max_backups': 10,
                'directories_to_backup': [
                    'components',
                    'agents',
                    'templates',
                    'custom_components'
                ],
                'files_to_backup': [
                    'config.py',
                    'omnimind.py',
                    'app.py',
                    'development_notes.md'
                ]
            }
            with open(self.backup_config, 'w') as f:
                json.dump(default_config, f, indent=4)

    def create_backup(self):
        """Create a new backup"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(self.backup_dir, f'backup_{timestamp}')
        
        try:
            # Create backup directory
            os.makedirs(backup_path)
            
            # Load config
            with open(self.backup_config, 'r') as f:
                config = json.load(f)
            
            # Backup directories
            for dir_name in config['directories_to_backup']:
                src_dir = os.path.join(self.base_dir, dir_name)
                if os.path.exists(src_dir):
                    dst_dir = os.path.join(backup_path, dir_name)
                    shutil.copytree(src_dir, dst_dir)
            
            # Backup individual files
            for file_name in config['files_to_backup']:
                src_file = os.path.join(self.base_dir, file_name)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, backup_path)
            
            # Update last backup time
            config['last_backup'] = timestamp
            with open(self.backup_config, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Cleanup old backups
            self._cleanup_old_backups(config['max_backups'])
            
            return True, f"Backup created successfully at {backup_path}"
        
        except Exception as e:
            return False, f"Backup failed: {str(e)}"

    def _cleanup_old_backups(self, max_backups):
        """Remove old backups if we exceed max_backups"""
        backups = sorted([d for d in os.listdir(self.backup_dir) 
                         if os.path.isdir(os.path.join(self.backup_dir, d))])
        
        while len(backups) > max_backups:
            oldest_backup = os.path.join(self.backup_dir, backups.pop(0))
            shutil.rmtree(oldest_backup)

    def restore_backup(self, backup_name):
        """Restore from a specific backup"""
        backup_path = os.path.join(self.backup_dir, backup_name)
        
        if not os.path.exists(backup_path):
            return False, "Backup not found"
        
        try:
            # Load config
            with open(self.backup_config, 'r') as f:
                config = json.load(f)
            
            # Restore directories
            for dir_name in config['directories_to_backup']:
                src_dir = os.path.join(backup_path, dir_name)
                if os.path.exists(src_dir):
                    dst_dir = os.path.join(self.base_dir, dir_name)
                    if os.path.exists(dst_dir):
                        shutil.rmtree(dst_dir)
                    shutil.copytree(src_dir, dst_dir)
            
            # Restore individual files
            for file_name in config['files_to_backup']:
                src_file = os.path.join(backup_path, file_name)
                if os.path.exists(src_file):
                    dst_file = os.path.join(self.base_dir, file_name)
                    shutil.copy2(src_file, dst_file)
            
            return True, f"Restored successfully from {backup_name}"
        
        except Exception as e:
            return False, f"Restore failed: {str(e)}"

    def list_backups(self):
        """List all available backups"""
        backups = []
        for backup in os.listdir(self.backup_dir):
            backup_path = os.path.join(self.backup_dir, backup)
            if os.path.isdir(backup_path):
                backup_size = sum(f.stat().st_size for f in Path(backup_path).rglob('*') if f.is_file())
                backups.append({
                    'name': backup,
                    'date': backup.split('_')[1],
                    'size_mb': round(backup_size / (1024 * 1024), 2)
                })
        return sorted(backups, key=lambda x: x['date'], reverse=True)

if __name__ == '__main__':
    # Example usage
    backup_system = BackupSystem()
    success, message = backup_system.create_backup()
    print(message)
    print("\nAvailable backups:")
    for backup in backup_system.list_backups():
        print(f"{backup['name']} - {backup['size_mb']}MB")
