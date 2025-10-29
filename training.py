"""
Training Loop with Regularization
=================================

This module handles model training with early stopping, regularization, and monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop early"""
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False

class TrainingMonitor:
    """Monitor training progress and metrics"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'learning_rate': []
        }
        
    def update(self, metrics: Dict[str, float]):
        """Update metrics history"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.metrics_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.metrics_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(self.metrics_history['train_mae'], label='Train MAE')
        axes[0, 1].plot(self.metrics_history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # RMSE
        axes[1, 0].plot(self.metrics_history['train_rmse'], label='Train RMSE')
        axes[1, 0].plot(self.metrics_history['val_rmse'], label='Val RMSE')
        axes[1, 0].set_title('Root Mean Square Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(self.metrics_history['learning_rate'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_metrics(self, save_path: str):
        """Save metrics to JSON file"""
        with open(save_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

class StressTrainer:
    """Main trainer class for stress prediction models"""
    
    def __init__(self, model: nn.Module, config, output_dir: str):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        
        # Initialize training components
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        self.monitor = TrainingMonitor(output_dir)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Training on device: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move to device
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            window_ids = batch['window_id']
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            # Compute loss
            loss = self.criterion(predictions, labels)
            
            # Add temporal smoothness regularization
            if len(window_ids) > 1:
                smoothness_loss = self._compute_temporal_smoothness_loss(predictions, window_ids)
                loss += 0.1 * smoothness_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                mae = torch.mean(torch.abs(predictions - labels)).item()
                rmse = torch.sqrt(torch.mean((predictions - labels) ** 2)).item()
                
                total_loss += loss.item()
                total_mae += mae
                total_rmse += rmse
                num_batches += 1
        
        return {
            'train_loss': total_loss / num_batches,
            'train_mae': total_mae / num_batches,
            'train_rmse': total_rmse / num_batches
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                predictions = self.model(features)
                
                # Compute loss
                loss = self.criterion(predictions, labels)
                
                # Compute metrics
                mae = torch.mean(torch.abs(predictions - labels)).item()
                rmse = torch.sqrt(torch.mean((predictions - labels) ** 2)).item()
                
                total_loss += loss.item()
                total_mae += mae
                total_rmse += rmse
                num_batches += 1
        
        if num_batches == 0:
            return {
                'val_loss': float('inf'),
                'val_mae': float('inf'),
                'val_rmse': float('inf')
            }
        
        return {
            'val_loss': total_loss / num_batches,
            'val_mae': total_mae / num_batches,
            'val_rmse': total_rmse / num_batches
        }
    
    def _compute_temporal_smoothness_loss(self, predictions: torch.Tensor, window_ids: List[str]) -> torch.Tensor:
        """Compute temporal smoothness loss"""
        if len(predictions) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Group predictions by participant
        participant_preds = {}
        for i, window_id in enumerate(window_ids):
            participant_id = window_id.split('_')[0]
            if participant_id not in participant_preds:
                participant_preds[participant_id] = []
            participant_preds[participant_id].append(predictions[i])
        
        smoothness_loss = 0.0
        for participant_id, preds in participant_preds.items():
            if len(preds) > 1:
                preds_tensor = torch.stack(preds)
                diff = torch.diff(preds_tensor.squeeze(), dim=0)
                smoothness_loss += torch.mean(diff ** 2)
        
        return smoothness_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Main training loop"""
        logger.info("Starting training...")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['val_loss'])
            
            # Record metrics
            epoch_metrics = {
                **train_metrics,
                **val_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.monitor.update(epoch_metrics)
            
            # Log progress
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val MAE: {val_metrics['val_mae']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping
            if self.early_stopping(val_metrics['val_loss'], self.model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        self.save_model()
        
        # Save metrics
        self.monitor.save_metrics(self.output_dir / "training_metrics.json")
        self.monitor.plot_metrics(self.output_dir / "training_plots.png")
        
        return self.monitor.metrics_history
    
    def save_model(self, model_name: str = "best_model.pt"):
        """Save the trained model"""
        model_path = self.output_dir / "models" / model_name
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {model_path}")

class GroupAwareTrainer(StressTrainer):
    """Trainer with group-aware regularization to reduce subgroup bias"""
    
    def __init__(self, model: nn.Module, config, output_dir: str, participants_df: pd.DataFrame):
        super().__init__(model, config, output_dir)
        self.participants_df = participants_df
        
        # Create participant to group mapping
        self.participant_groups = {}
        for _, row in participants_df.iterrows():
            self.participant_groups[str(row['participant_id'])] = row['study_group']
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with group-aware regularization"""
        self.model.train()
        
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move to device
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            window_ids = batch['window_id']
            participant_ids = batch['participant_id']
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            # Compute main loss
            loss = self.criterion(predictions, labels)
            
            # Add temporal smoothness regularization
            if len(window_ids) > 1:
                smoothness_loss = self._compute_temporal_smoothness_loss(predictions, window_ids)
                loss += 0.1 * smoothness_loss
            
            # Add group-aware regularization
            group_penalty = self._compute_group_penalty(predictions, labels, participant_ids)
            loss += 0.05 * group_penalty
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                mae = torch.mean(torch.abs(predictions - labels)).item()
                rmse = torch.sqrt(torch.mean((predictions - labels) ** 2)).item()
                
                total_loss += loss.item()
                total_mae += mae
                total_rmse += rmse
                num_batches += 1
        
        return {
            'train_loss': total_loss / num_batches,
            'train_mae': total_mae / num_batches,
            'train_rmse': total_rmse / num_batches
        }
    
    def _compute_group_penalty(self, predictions: torch.Tensor, labels: torch.Tensor, 
                             participant_ids: List[str]) -> torch.Tensor:
        """Compute group-aware penalty to reduce subgroup bias"""
        # Group predictions by study group
        group_errors = {}
        
        for i, participant_id in enumerate(participant_ids):
            group = self.participant_groups.get(participant_id, 'unknown')
            error = torch.abs(predictions[i] - labels[i])
            
            if group not in group_errors:
                group_errors[group] = []
            group_errors[group].append(error)
        
        # Compute variance of errors across groups
        if len(group_errors) > 1:
            group_mean_errors = []
            for group, errors in group_errors.items():
                if errors:
                    group_mean_errors.append(torch.stack(errors).mean())
            
            if len(group_mean_errors) > 1:
                group_mean_errors = torch.stack(group_mean_errors)
                group_variance = torch.var(group_mean_errors)
                return group_variance
        
        return torch.tensor(0.0, device=predictions.device)
