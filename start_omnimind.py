import os
import sys
import subprocess
import time
import webbrowser
import tkinter as tk
from tkinter import ttk
import threading
import psutil

class OmniMindLauncher:
    def __init__(self):
        self.processes = {}
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("OmniMind Launcher")
        self.root.geometry("600x400")

        # Status indicators
        self.status_frame = ttk.LabelFrame(self.root, text="System Status")
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)

        # Status indicators
        self.indicators = {
            'web': ttk.Label(self.status_frame, text="Web Interface: Stopped", foreground="red"),
            'manager': ttk.Label(self.status_frame, text="Project Manager: Stopped", foreground="red")
        }
        for indicator in self.indicators.values():
            indicator.pack(anchor=tk.W, padx=5, pady=2)

        # Control buttons
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(self.button_frame, text="Start All", command=self.start_all)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(self.button_frame, text="Stop All", command=self.stop_all)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Log display
        self.log_frame = ttk.LabelFrame(self.root, text="System Log")
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = tk.Text(self.log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Quick access buttons
        self.quick_frame = ttk.LabelFrame(self.root, text="Quick Access")
        self.quick_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(self.quick_frame, text="Open Web Interface", 
                  command=lambda: webbrowser.open('https://127.0.0.1:5000')).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(self.quick_frame, text="Open Documentation", 
                  command=self.open_docs).pack(side=tk.LEFT, padx=5, pady=5)

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def start_web_interface(self):
        try:
            process = subprocess.Popen([sys.executable, 'app.py'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
            self.processes['web'] = process
            self.indicators['web'].config(text="Web Interface: Running", foreground="green")
            self.log("Web interface started")
            
            # Wait for startup
            time.sleep(2)
            webbrowser.open('https://127.0.0.1:5000')
        except Exception as e:
            self.log(f"Error starting web interface: {str(e)}")

    def start_project_manager(self):
        try:
            process = subprocess.Popen([sys.executable, 'project_manager.py'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
            self.processes['manager'] = process
            self.indicators['manager'].config(text="Project Manager: Running", foreground="green")
            self.log("Project manager started")
        except Exception as e:
            self.log(f"Error starting project manager: {str(e)}")

    def start_all(self):
        self.log("Starting all systems...")
        threading.Thread(target=self.start_web_interface, daemon=True).start()
        threading.Thread(target=self.start_project_manager, daemon=True).start()

    def stop_all(self):
        self.log("Stopping all systems...")
        for name, process in self.processes.items():
            try:
                # Get the process and all its children
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                
                # Terminate children first
                for child in children:
                    child.terminate()
                psutil.wait_procs(children, timeout=3)
                
                # Terminate parent
                parent.terminate()
                parent.wait(3)
                
                self.indicators[name].config(text=f"{name.title()}: Stopped", foreground="red")
                self.log(f"Stopped {name}")
            except Exception as e:
                self.log(f"Error stopping {name}: {str(e)}")
        
        self.processes.clear()

    def open_docs(self):
        try:
            subprocess.Popen([sys.executable, '-m', 'markdown_viewer', 'development_notes.md'])
            self.log("Opened documentation")
        except Exception as e:
            self.log(f"Error opening documentation: {str(e)}")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        if self.processes:
            self.stop_all()
        self.root.destroy()

if __name__ == "__main__":
    launcher = OmniMindLauncher()
    launcher.run()
