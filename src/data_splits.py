"""
Data Splits and Dataset Classes
===============================

This module handles participant-based train/val/test splits and PyTorch dataset creation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class StressDataset(Dataset):
    """PyTorch Dataset for stress prediction"""
    
    def __init__(self, windows: List[Dict], scaler: Optional[StandardScaler] = None, fit_scaler: bool = False):
        """
        Initialize dataset
        
        Args:
            windows: List of window dictionaries
            scaler: Fitted StandardScaler for normalization
            fit_scaler: Whether to fit the scaler on this data
        """
        self.windows = windows
        self.scaler = scaler
        
        if fit_scaler and scaler is None:
            self.scaler = self._fit_scaler()
        elif scaler is not None:
            self._apply_scaling()
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        # Extract features and label
        features = torch.FloatTensor(window['features'])
        label = torch.FloatTensor([window['stress_label']])
        
        # Add participant ID for group-aware training
        participant_id = window['participant_id']
        
        return {
            'features': features,
            'label': label,
            'participant_id': participant_id,
            'window_id': window['window_id']
        }
    
    def _fit_scaler(self) -> StandardScaler:
        """Fit StandardScaler on all features"""
        logger.info("Fitting StandardScaler on training data...")
        
        # Collect all features
        all_features = []
        for window in self.windows:
            features = window['features'].flatten()  # Flatten to 1D
            all_features.append(features)
        
        all_features = np.array(all_features)
        
        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(all_features)
        
        # Apply scaling
        self._apply_scaling()
        
        return scaler
    
    def _apply_scaling(self):
        """Apply fitted scaler to all features"""
        if self.scaler is None:
            return
        
        logger.info("Applying feature scaling...")
        
        for window in self.windows:
            original_shape = window['features'].shape
            features_flat = window['features'].flatten()
            features_scaled = self.scaler.transform(features_flat.reshape(1, -1)).flatten()
            window['features'] = features_scaled.reshape(original_shape)

class DataSplitter:
    """Handles participant-based train/val/test splits"""
    
    def __init__(self, config):
        self.config = config
        
    def create_splits(self, all_windows: List[Dict], participants_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create participant-based train/val/test splits
        
        Args:
            all_windows: All windows from all participants
            participants_df: Participant metadata DataFrame
            
        Returns:
            Tuple of (train_windows, val_windows, test_windows)
        """
        # Group windows by participant
        windows_by_participant = {}
        for window in all_windows:
            pid = window['participant_id']
            if pid not in windows_by_participant:
                windows_by_participant[pid] = []
            windows_by_participant[pid].append(window)
        
        # Get participant splits from metadata
        train_participants = participants_df[participants_df['recommended_split'] == 'train']['participant_id'].astype(str).tolist()
        val_participants = participants_df[participants_df['recommended_split'] == 'val']['participant_id'].astype(str).tolist()
        test_participants = participants_df[participants_df['recommended_split'] == 'test']['participant_id'].astype(str).tolist()
        
        # Create splits
        train_windows = []
        val_windows = []
        test_windows = []
        
        for pid in train_participants:
            if pid in windows_by_participant:
                train_windows.extend(windows_by_participant[pid])
        
        for pid in val_participants:
            if pid in windows_by_participant:
                val_windows.extend(windows_by_participant[pid])
        
        for pid in test_participants:
            if pid in windows_by_participant:
                test_windows.extend(windows_by_participant[pid])
        
        logger.info(f"Created splits - Train: {len(train_windows)}, Val: {len(val_windows)}, Test: {len(test_windows)}")
        
        return train_windows, val_windows, test_windows
    
    def create_dataloaders(self, train_windows: List[Dict], val_windows: List[Dict], 
                          test_windows: List[Dict]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train/val/test sets
        
        Args:
            train_windows: Training windows
            val_windows: Validation windows  
            test_windows: Test windows
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = StressDataset(train_windows, fit_scaler=True)
        val_dataset = StressDataset(val_windows, scaler=train_dataset.scaler)
        test_dataset = StressDataset(test_windows, scaler=train_dataset.scaler)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_split_statistics(self, train_windows: List[Dict], val_windows: List[Dict], 
                            test_windows: List[Dict]) -> Dict[str, Dict]:
        """Get statistics for each split"""
        
        def get_split_stats(windows):
            if not windows:
                return {}
            
            stress_labels = [w['stress_label'] for w in windows if w['stress_label'] is not None]
            participants = list(set(w['participant_id'] for w in windows))
            
            return {
                'num_windows': len(windows),
                'num_participants': len(participants),
                'windows_with_labels': len(stress_labels),
                'label_coverage': len(stress_labels) / len(windows) if windows else 0,
                'stress_mean': np.mean(stress_labels) if stress_labels else np.nan,
                'stress_std': np.std(stress_labels) if stress_labels else np.nan,
                'stress_min': np.min(stress_labels) if stress_labels else np.nan,
                'stress_max': np.max(stress_labels) if stress_labels else np.nan
            }
        
        return {
            'train': get_split_stats(train_windows),
            'val': get_split_stats(val_windows),
            'test': get_split_stats(test_windows)
        }

class BalancedSampler:
    """Custom sampler for balanced stress level sampling"""
    
    def __init__(self, dataset: StressDataset, stress_threshold: float = 0.5):
        self.dataset = dataset
        self.stress_threshold = stress_threshold
        
        # Create balanced indices
        self.indices = self._create_balanced_indices()
    
    def _create_balanced_indices(self) -> List[int]:
        """Create balanced indices for high/low stress samples"""
        high_stress_indices = []
        low_stress_indices = []
        
        for idx, window in enumerate(self.dataset.windows):
            if window['stress_label'] is not None:
                if window['stress_label'] >= self.stress_threshold:
                    high_stress_indices.append(idx)
                else:
                    low_stress_indices.append(idx)
        
        # Balance the classes
        min_class_size = min(len(high_stress_indices), len(low_stress_indices))
        
        balanced_indices = []
        balanced_indices.extend(high_stress_indices[:min_class_size])
        balanced_indices.extend(low_stress_indices[:min_class_size])
        
        return balanced_indices
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)
