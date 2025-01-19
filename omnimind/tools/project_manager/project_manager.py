import os
import sys
import datetime
import subprocess
from backup_system import BackupSystem
from version_control import LocalVersionControl
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import markdown
import webbrowser
from tkhtmlview import HTMLLabel

class ProjectManager:
    def __init__(self):
        self.backup_system = BackupSystem()
        self.version_control = LocalVersionControl()
        self.notes_file = 'development_notes.md'
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("OmniMind Project Manager")
        self.root.geometry("1200x800")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Create tabs
        self.backup_tab = ttk.Frame(self.notebook)
        self.version_tab = ttk.Frame(self.notebook)
        self.notes_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.backup_tab, text='Backup System')
        self.notebook.add(self.version_tab, text='Version Control')
        self.notebook.add(self.notes_tab, text='Development Notes')
        
        # Setup each tab
        self.setup_backup_tab()
        self.setup_version_tab()
        self.setup_notes_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.update_status("Ready")

    def setup_backup_tab(self):
        # Left frame for controls
        left_frame = ttk.Frame(self.backup_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create backup button
        ttk.Button(left_frame, text="Create Backup", 
                  command=self.create_backup).pack(fill=tk.X, pady=5)
        
        # Restore backup button
        ttk.Button(left_frame, text="Restore Selected", 
                  command=self.restore_backup).pack(fill=tk.X, pady=5)
        
        # Right frame for backup list
        right_frame = ttk.Frame(self.backup_tab)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Backup list
        columns = ('Name', 'Date', 'Size')
        self.backup_tree = ttk.Treeview(right_frame, columns=columns, show='headings')
        
        for col in columns:
            self.backup_tree.heading(col, text=col)
            self.backup_tree.column(col, width=100)
        
        self.backup_tree.pack(fill=tk.BOTH, expand=True)
        self.refresh_backup_list()

    def setup_version_tab(self):
        # Left frame for controls
        left_frame = ttk.Frame(self.version_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Version message entry
        ttk.Label(left_frame, text="Version Message:").pack(fill=tk.X, pady=2)
        self.version_message = ttk.Entry(left_frame)
        self.version_message.pack(fill=tk.X, pady=5)
        
        # Create version button
        ttk.Button(left_frame, text="Create Version", 
                  command=self.create_version).pack(fill=tk.X, pady=5)
        
        # Restore version button
        ttk.Button(left_frame, text="Restore Selected", 
                  command=self.restore_version).pack(fill=tk.X, pady=5)
        
        # Right frame for version list
        right_frame = ttk.Frame(self.version_tab)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Version list
        columns = ('ID', 'Message', 'Date', 'Current')
        self.version_tree = ttk.Treeview(right_frame, columns=columns, show='headings')
        
        for col in columns:
            self.version_tree.heading(col, text=col)
            self.version_tree.column(col, width=100)
        
        self.version_tree.pack(fill=tk.BOTH, expand=True)
        self.refresh_version_list()

    def setup_notes_tab(self):
        # Create frames
        control_frame = ttk.Frame(self.notes_tab)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        editor_frame = ttk.Frame(self.notes_tab)
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons
        ttk.Button(control_frame, text="Save", 
                  command=self.save_notes).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Preview", 
                  command=self.preview_notes).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Insert Template", 
                  command=self.insert_template).pack(side=tk.LEFT, padx=5)
        
        # Create editor
        self.notes_editor = scrolledtext.ScrolledText(editor_frame, wrap=tk.WORD,
                                                    width=80, height=30)
        self.notes_editor.pack(fill=tk.BOTH, expand=True)
        
        # Load current notes
        self.load_notes()

    def create_backup(self):
        success, message = self.backup_system.create_backup()
        self.update_status(message)
        self.refresh_backup_list()

    def restore_backup(self):
        selection = self.backup_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a backup to restore")
            return
        
        backup_name = self.backup_tree.item(selection[0])['values'][0]
        if messagebox.askyesno("Confirm Restore", 
                             f"Are you sure you want to restore backup {backup_name}?"):
            success, message = self.backup_system.restore_backup(backup_name)
            self.update_status(message)

    def create_version(self):
        message = self.version_message.get()
        if not message:
            messagebox.showwarning("Warning", "Please enter a version message")
            return
        
        success, message = self.version_control.create_version(message)
        self.update_status(message)
        self.version_message.delete(0, tk.END)
        self.refresh_version_list()

    def restore_version(self):
        selection = self.version_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a version to restore")
            return
        
        version_id = self.version_tree.item(selection[0])['values'][0]
        if messagebox.askyesno("Confirm Restore", 
                             f"Are you sure you want to restore version {version_id}?"):
            success, message = self.version_control.restore_version(version_id)
            self.update_status(message)
            self.refresh_version_list()

    def load_notes(self):
        try:
            with open(self.notes_file, 'r') as f:
                self.notes_editor.delete('1.0', tk.END)
                self.notes_editor.insert('1.0', f.read())
        except Exception as e:
            self.update_status(f"Error loading notes: {str(e)}")

    def save_notes(self):
        try:
            with open(self.notes_file, 'w') as f:
                f.write(self.notes_editor.get('1.0', tk.END))
            self.update_status("Notes saved successfully")
        except Exception as e:
            self.update_status(f"Error saving notes: {str(e)}")

    def preview_notes(self):
        # Create preview window
        preview = tk.Toplevel(self.root)
        preview.title("Markdown Preview")
        preview.geometry("800x600")
        
        # Convert markdown to HTML
        md_text = self.notes_editor.get('1.0', tk.END)
        html = markdown.markdown(md_text)
        
        # Display HTML
        preview_html = HTMLLabel(preview, html=html)
        preview_html.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def insert_template(self):
        template = """## New Section
### Description
[Add description here]

### Details
- Point 1
- Point 2
- Point 3

### Status
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

---"""
        self.notes_editor.insert(tk.INSERT, template)

    def refresh_backup_list(self):
        # Clear current items
        for item in self.backup_tree.get_children():
            self.backup_tree.delete(item)
        
        # Add backups to tree
        for backup in self.backup_system.list_backups():
            self.backup_tree.insert('', tk.END, values=(
                backup['name'],
                backup['date'],
                f"{backup['size_mb']}MB"
            ))

    def refresh_version_list(self):
        # Clear current items
        for item in self.version_tree.get_children():
            self.version_tree.delete(item)
        
        # Add versions to tree
        for version in self.version_control.list_versions():
            self.version_tree.insert('', tk.END, values=(
                version['id'],
                version['message'],
                version['timestamp'],
                "Yes" if version['is_current'] else "No"
            ))

    def update_status(self, message):
        self.status_var.set(f"{datetime.datetime.now().strftime('%H:%M:%S')}: {message}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    manager = ProjectManager()
    manager.run()
