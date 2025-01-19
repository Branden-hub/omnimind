import importlib
import sys
import cv2
import mss
import numpy as np
import requests
from PIL import Image
from bs4 import BeautifulSoup
from typing import Dict, Any, List

class ExtensionManager:
    def __init__(self):
        self.loaded_extensions: Dict[str, Any] = {}
        self.screen_capture = mss.mss()
        self.camera = None
        self._setup_camera()
        
    def _setup_camera(self):
        """Initialize camera access"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("Warning: Could not access camera")
        except Exception as e:
            print(f"Camera initialization error: {e}")
            
    def load_extension(self, extension_path: str) -> bool:
        """Dynamically load an extension from a Python file"""
        try:
            spec = importlib.util.spec_from_file_location(
                f"extension_{len(self.loaded_extensions)}", 
                extension_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            
            # Store the loaded extension
            extension_name = getattr(module, "EXTENSION_NAME", extension_path)
            self.loaded_extensions[extension_name] = module
            return True
        except Exception as e:
            print(f"Failed to load extension {extension_path}: {e}")
            return False
            
    def get_screen_content(self) -> np.ndarray:
        """Capture current screen content"""
        try:
            screen = self.screen_capture.grab(self.screen_capture.monitors[0])
            return np.array(Image.frombytes('RGB', screen.size, screen.rgb))
        except Exception as e:
            print(f"Screen capture error: {e}")
            return None
            
    def get_camera_frame(self) -> np.ndarray:
        """Capture current camera frame"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None
        
    def search_web(self, query: str) -> List[Dict[str, str]]:
        """Search the web for information"""
        try:
            # Use DuckDuckGo as it doesn't require API key
            url = f"https://duckduckgo.com/html/?q={query}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='result'):
                title = result.find('h2').text if result.find('h2') else ''
                snippet = result.find('a', class_='result__snippet').text if result.find('a', class_='result__snippet') else ''
                link = result.find('a', class_='result__url').get('href') if result.find('a', class_='result__url') else ''
                
                results.append({
                    'title': title,
                    'snippet': snippet,
                    'link': link
                })
                
            return results
        except Exception as e:
            print(f"Web search error: {e}")
            return []
            
    def augment_data(self, data: Any) -> Any:
        """Augment data using loaded extensions"""
        augmented_data = data
        for extension in self.loaded_extensions.values():
            if hasattr(extension, 'augment_data'):
                try:
                    augmented_data = extension.augment_data(augmented_data)
                except Exception as e:
                    print(f"Data augmentation error in extension: {e}")
        return augmented_data
        
    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.release()
        self.screen_capture.close()
