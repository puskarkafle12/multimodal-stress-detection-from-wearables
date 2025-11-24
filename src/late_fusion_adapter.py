"""
Late Fusion Data Adapter
========================

Converts concatenated features to dictionary format for late fusion models.
"""

import torch
import numpy as np
from typing import Dict, List
from torch.utils.data import Dataset

# Modality order matches windowing.py
MODALITY_ORDER = ['heart_rate', 'sleep', 'cgm', 'oxygen_saturation', 'respiratory_rate']
MODALITY_DIMS = {mod: 2 for mod in MODALITY_ORDER}  # Each has value + mask

def split_features_to_modalities(features: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Split concatenated features into separate modality tensors
    
    Args:
        features: Concatenated features [batch_size, window_length, total_features]
                 where total_features = num_modalities * 2 (value + mask for each)
    
    Returns:
        Dictionary mapping modality names to their feature tensors
        Each tensor shape: [batch_size, window_length, 2] (value + mask)
    """
    batch_size, window_length, total_features = features.shape
    
    # Expected: 5 modalities * 2 channels = 10 features
    expected_features = len(MODALITY_ORDER) * 2
    if total_features != expected_features:
        raise ValueError(f"Expected {expected_features} features (5 modalities * 2), got {total_features}")
    
    modality_dict = {}
    feature_idx = 0
    
    for modality in MODALITY_ORDER:
        # Extract 2 features (value + mask) for this modality
        modality_features = features[:, :, feature_idx:feature_idx + 2]
        modality_dict[modality] = modality_features
        feature_idx += 2
    
    return modality_dict

def late_fusion_collate_fn(batch):
    """
    Custom collate function for late fusion batches
    Handles dict features properly when batching
    """
    # Extract features (dicts) and other fields
    features_list = [item['features'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    participant_ids = [item['participant_id'] for item in batch]
    window_ids = [item['window_id'] for item in batch]
    
    # Stack features for each modality
    batched_features = {}
    for modality in MODALITY_ORDER:
        modality_tensors = [feat[modality] for feat in features_list]
        batched_features[modality] = torch.stack(modality_tensors)
    
    return {
        'features': batched_features,
        'label': labels,
        'participant_id': participant_ids,
        'window_id': window_ids
    }

class LateFusionDatasetAdapter(Dataset):
    """Adapter to convert concatenated features to late fusion format"""
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Split concatenated features into modalities
        # sample['features'] is [window_length, total_features]
        # Need to add batch dimension temporarily
        features = sample['features'].unsqueeze(0)  # [1, window_length, total_features]
        features_dict = split_features_to_modalities(features)
        # Remove batch dimension
        features_dict = {k: v.squeeze(0) for k, v in features_dict.items()}
        
        return {
            'features': features_dict,  # Now a dict instead of tensor
            'label': sample['label'],
            'participant_id': sample['participant_id'],
            'window_id': sample['window_id']
        }

