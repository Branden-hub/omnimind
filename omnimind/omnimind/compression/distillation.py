import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json

class DistillationManager:
    """Base class for knowledge distillation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.distillation_config = {}
        self.stats = {
            'teacher_size': 0,
            'student_size': 0,
            'compression_ratio': 1.0
        }
    
    def _get_model_size(self, model: torch.nn.Module) -> int:
        """Get model size in bytes."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size
    
    def _update_stats(self,
                     teacher: torch.nn.Module,
                     student: torch.nn.Module) -> None:
        """Update distillation statistics."""
        teacher_size = self._get_model_size(teacher)
        student_size = self._get_model_size(student)
        
        self.stats.update({
            'teacher_size': teacher_size,
            'student_size': student_size,
            'compression_ratio': teacher_size / student_size
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get distillation statistics."""
        return self.stats.copy()
    
    def save_student_model(self,
                          model: torch.nn.Module,
                          path: Union[str, Path]) -> None:
        """Save distilled student model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'state_dict': model.state_dict(),
            'stats': self.stats,
            'config': self.distillation_config
        }, path)
    
    def load_student_model(self,
                          model: torch.nn.Module,
                          path: Union[str, Path]) -> torch.nn.Module:
        """Load distilled student model."""
        path = Path(path)
        
        # Load checkpoint
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        self.stats = checkpoint['stats']
        self.distillation_config = checkpoint['config']
        
        return model

class KnowledgeDistiller(DistillationManager):
    """Handles knowledge distillation training."""
    
    def __init__(self,
                 temperature: float = 2.0,
                 alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(self,
                         student_logits: torch.Tensor,
                         teacher_logits: torch.Tensor,
                         labels: torch.Tensor,
                         temperature: Optional[float] = None) -> torch.Tensor:
        """Calculate distillation loss."""
        T = temperature or self.temperature
        
        # Soften probabilities and calculate distillation loss
        soft_targets = F.softmax(teacher_logits / T, dim=1)
        soft_prob = F.log_softmax(student_logits / T, dim=1)
        distillation_loss = -(soft_targets * soft_prob).sum(dim=1).mean()
        
        # Calculate standard cross entropy with true labels
        student_loss = F.cross_entropy(student_logits, labels)
        
        # Combine losses
        loss = (self.alpha * student_loss +
                (1 - self.alpha) * (T ** 2) * distillation_loss)
        
        return loss
    
    def train_step(self,
                   teacher: torch.nn.Module,
                   student: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   inputs: torch.Tensor,
                   labels: torch.Tensor) -> Dict[str, float]:
        """Perform one step of distillation training."""
        teacher.eval()
        student.train()
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        
        # Get student predictions
        student_logits = student(inputs)
        
        # Calculate loss
        loss = self.distillation_loss(
            student_logits,
            teacher_logits,
            labels
        )
        
        # Update student
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'student_acc': (student_logits.argmax(dim=1) == labels).float().mean().item()
        }
    
    def train_epoch(self,
                    teacher: torch.nn.Module,
                    student: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    dataloader: torch.utils.data.DataLoader,
                    device: torch.device) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_stats = {
            'loss': 0.0,
            'student_acc': 0.0
        }
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Perform training step
            step_stats = self.train_step(
                teacher,
                student,
                optimizer,
                inputs,
                labels
            )
            
            # Update epoch statistics
            for key in epoch_stats:
                epoch_stats[key] += step_stats[key]
        
        # Average statistics
        for key in epoch_stats:
            epoch_stats[key] /= len(dataloader)
        
        return epoch_stats
    
    def distill_knowledge(self,
                         teacher: torch.nn.Module,
                         student: torch.nn.Module,
                         train_loader: torch.utils.data.DataLoader,
                         val_loader: Optional[torch.utils.data.DataLoader] = None,
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         num_epochs: int = 100,
                         device: Optional[torch.device] = None) -> torch.nn.Module:
        """Perform knowledge distillation."""
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        teacher = teacher.to(device)
        student = student.to(device)
        
        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(student.parameters())
        
        # Store configuration
        self.distillation_config = {
            'temperature': self.temperature,
            'alpha': self.alpha,
            'num_epochs': num_epochs
        }
        
        # Update statistics
        self._update_stats(teacher, student)
        
        # Training loop
        best_val_acc = 0.0
        best_student = None
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_stats = self.train_epoch(
                teacher,
                student,
                optimizer,
                train_loader,
                device
            )
            
            # Validate if validation loader provided
            if val_loader is not None:
                val_stats = self.validate(
                    student,
                    val_loader,
                    device
                )
                
                # Save best model
                if val_stats['accuracy'] > best_val_acc:
                    best_val_acc = val_stats['accuracy']
                    best_student = student.state_dict()
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_stats['loss']:.4f}, "
                    f"Train Acc: {train_stats['student_acc']:.4f}, "
                    f"Val Acc: {val_stats['accuracy']:.4f}"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_stats['loss']:.4f}, "
                    f"Train Acc: {train_stats['student_acc']:.4f}"
                )
        
        # Load best model if validation was performed
        if best_student is not None:
            student.load_state_dict(best_student)
        
        return student
    
    def validate(self,
                model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                device: torch.device) -> Dict[str, float]:
        """Validate model performance."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'accuracy': correct / total
        }
