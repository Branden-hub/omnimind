"""
OmniMind: Unified Interface
"""

import tkinter as tk
from tkinter import ttk
import threading
import logging
from typing import Optional

from .omnimind import OmniMind
from .interactions import OmniMindInteractions

class UnifiedInterface:
    def __init__(self, config_path: Optional[str] = None):
        # Initialize OmniMind framework
        self.omnimind = OmniMind(config_path)
        
        # Initialize interactions
        self.interactions = OmniMindInteractions()
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def setup_gui(self):
        """Setup the main GUI window."""
        self.root = tk.Tk()
        self.root.title("OmniMind Unified Interface")
        self.root.geometry("800x600")

        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Create tabs
        self.setup_control_tab()
        self.setup_model_tab()
        self.setup_interaction_tab()

    def setup_control_tab(self):
        """Setup the main control tab."""
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="Control Panel")

        # Status indicators
        status_frame = ttk.LabelFrame(control_frame, text="System Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        self.status_labels = {
            'system': ttk.Label(status_frame, text="System: Active", foreground="green"),
            'voice': ttk.Label(status_frame, text="Voice: Inactive", foreground="red"),
            'camera': ttk.Label(status_frame, text="Camera: Inactive", foreground="red"),
            'screen': ttk.Label(status_frame, text="Screen: Inactive", foreground="red")
        }
        for label in self.status_labels.values():
            label.pack(anchor=tk.W, padx=5, pady=2)

    def setup_model_tab(self):
        """Setup the model management tab."""
        model_frame = ttk.Frame(self.notebook)
        self.notebook.add(model_frame, text="Model Management")

        # Model controls
        controls_frame = ttk.LabelFrame(model_frame, text="Model Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(controls_frame, text="Load Model", 
                  command=self.omnimind.model_registry.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Optimize Model", 
                  command=self.optimize_model).pack(side=tk.LEFT, padx=5)

    def setup_interaction_tab(self):
        """Setup the interaction control tab."""
        interaction_frame = ttk.Frame(self.notebook)
        self.notebook.add(interaction_frame, text="Interactions")

        # Interaction controls
        controls_frame = ttk.LabelFrame(interaction_frame, text="Interaction Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(controls_frame, text="Start Voice", 
                  command=self.start_voice).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Start Camera", 
                  command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Start Screen", 
                  command=self.start_screen).pack(side=tk.LEFT, padx=5)

    def start_voice(self):
        """Start voice interaction."""
        threading.Thread(target=self.interactions.speech_to_text, daemon=True).start()
        self.status_labels['voice'].config(text="Voice: Active", foreground="green")

    def start_camera(self):
        """Start camera."""
        self.interactions.start_camera_thread()
        self.status_labels['camera'].config(text="Camera: Active", foreground="green")

    def start_screen(self):
        """Start screen capture."""
        self.interactions.start_screen_thread()
        self.status_labels['screen'].config(text="Screen: Active", foreground="green")

    def optimize_model(self):
        """Optimize the current model."""
        try:
            self.omnimind.script_compiler.compile()
            self.omnimind.onnx_exporter.export()
            self.logger.info("Model optimization completed successfully")
        except Exception as e:
            self.logger.error(f"Error during model optimization: {str(e)}")

    def run(self):
        """Start the unified interface."""
        try:
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
        finally:
            # Cleanup
            if hasattr(self.interactions, 'camera_running'):
                self.interactions.camera_running = False
            if hasattr(self.interactions, 'screen_running'):
                self.screen_running = False
