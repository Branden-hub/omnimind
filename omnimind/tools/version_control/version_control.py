import os
import json
import shutil
import hashlib
import datetime
from pathlib import Path

class LocalVersionControl:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.versions_dir = os.path.join(self.base_dir, '.versions')
        self.version_db = os.path.join(self.versions_dir, 'version_db.json')
        self.initialize_system()

    def initialize_system(self):
        """Initialize the version control system"""
        if not os.path.exists(self.versions_dir):
            os.makedirs(self.versions_dir)
        
        if not os.path.exists(self.version_db):
            initial_db = {
                'versions': [],
                'current_version': None,
                'tracked_files': [
                    '*.py',
                    '*.html',
                    '*.md',
                    'templates/*',
                    'components/*'
                ],
                'ignored_patterns': [
                    '__pycache__',
                    '*.pyc',
                    '.versions/*',
                    'backups/*'
                ]
            }
            with open(self.version_db, 'w') as f:
                json.dump(initial_db, f, indent=4)

    def create_version(self, message):
        """Create a new version with the current state"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        version_path = os.path.join(self.versions_dir, timestamp)
        
        try:
            # Create version directory
            os.makedirs(version_path)
            
            # Load version database
            with open(self.version_db, 'r') as f:
                db = json.load(f)
            
            # Track files
            tracked_files = self._get_tracked_files()
            file_hashes = {}
            
            for file_path in tracked_files:
                rel_path = os.path.relpath(file_path, self.base_dir)
                dst_path = os.path.join(version_path, rel_path)
                
                # Create necessary directories
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # Copy file and compute hash
                shutil.copy2(file_path, dst_path)
                file_hashes[rel_path] = self._compute_file_hash(file_path)
            
            # Update version database
            version_info = {
                'id': timestamp,
                'message': message,
                'timestamp': timestamp,
                'file_hashes': file_hashes
            }
            
            db['versions'].append(version_info)
            db['current_version'] = timestamp
            
            with open(self.version_db, 'w') as f:
                json.dump(db, f, indent=4)
            
            return True, f"Version {timestamp} created successfully"
        
        except Exception as e:
            return False, f"Failed to create version: {str(e)}"

    def restore_version(self, version_id):
        """Restore to a specific version"""
        version_path = os.path.join(self.versions_dir, version_id)
        
        if not os.path.exists(version_path):
            return False, "Version not found"
        
        try:
            # Load version database
            with open(self.version_db, 'r') as f:
                db = json.load(f)
            
            # Find version info
            version_info = next((v for v in db['versions'] if v['id'] == version_id), None)
            if not version_info:
                return False, "Version information not found"
            
            # Restore files
            for rel_path, file_hash in version_info['file_hashes'].items():
                src_path = os.path.join(version_path, rel_path)
                dst_path = os.path.join(self.base_dir, rel_path)
                
                if os.path.exists(src_path):
                    # Create necessary directories
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path)
            
            db['current_version'] = version_id
            with open(self.version_db, 'w') as f:
                json.dump(db, f, indent=4)
            
            return True, f"Restored to version {version_id}"
        
        except Exception as e:
            return False, f"Failed to restore version: {str(e)}"

    def list_versions(self):
        """List all versions with their messages"""
        try:
            with open(self.version_db, 'r') as f:
                db = json.load(f)
            
            versions = []
            for version in db['versions']:
                versions.append({
                    'id': version['id'],
                    'message': version['message'],
                    'timestamp': version['timestamp'],
                    'is_current': version['id'] == db['current_version']
                })
            
            return sorted(versions, key=lambda x: x['timestamp'], reverse=True)
        
        except Exception:
            return []

    def _get_tracked_files(self):
        """Get list of tracked files based on patterns"""
        with open(self.version_db, 'r') as f:
            db = json.load(f)
        
        tracked_files = []
        for pattern in db['tracked_files']:
            if '*' in pattern:
                # Handle wildcard patterns
                directory = os.path.dirname(pattern) or '.'
                file_pattern = os.path.basename(pattern)
                
                for root, _, files in os.walk(os.path.join(self.base_dir, directory)):
                    for file in files:
                        if self._matches_pattern(file, file_pattern):
                            file_path = os.path.join(root, file)
                            if not self._is_ignored(file_path):
                                tracked_files.append(file_path)
            else:
                # Handle exact paths
                file_path = os.path.join(self.base_dir, pattern)
                if os.path.exists(file_path) and not self._is_ignored(file_path):
                    tracked_files.append(file_path)
        
        return tracked_files

    def _is_ignored(self, file_path):
        """Check if file should be ignored"""
        with open(self.version_db, 'r') as f:
            db = json.load(f)
        
        rel_path = os.path.relpath(file_path, self.base_dir)
        return any(self._matches_pattern(rel_path, pattern) for pattern in db['ignored_patterns'])

    @staticmethod
    def _matches_pattern(filename, pattern):
        """Check if filename matches the pattern"""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)

    @staticmethod
    def _compute_file_hash(file_path):
        """Compute SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

if __name__ == '__main__':
    # Example usage
    vc = LocalVersionControl()
    success, message = vc.create_version("Initial version")
    print(message)
    print("\nAvailable versions:")
    for version in vc.list_versions():
        current = "(current)" if version['is_current'] else ""
        print(f"{version['id']} {current} - {version['message']}")
