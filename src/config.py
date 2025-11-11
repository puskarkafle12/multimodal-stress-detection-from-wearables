"""
AI-READI Wearable Stress Prediction Pipeline
==========================================

A comprehensive pipeline for preprocessing wearable data and predicting stress levels
from multimodal physiological signals.

Pipeline Overview:
1. Data Ingestion & Harmonization
2. Windowing & Label Alignment  
3. Participant-based Train/Val/Test Splits
4. Modeling (CNN/Transformer architectures)
5. Training with Regularization
6. Inference & Participant-level Aggregation
7. Evaluation & Interpretability
8. Robustness Testing

Author: AI Assistant
Date: 2024
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

@dataclass
class PipelineConfig:
    """Configuration for the stress prediction pipeline"""
    
    # Data paths
    data_root: str = "/Users/puskarkafle/Documents/Research/AI-READI"
    output_dir: str = "/Users/puskarkafle/Documents/Research/stress_pipeline_output"
    
    # Windowing parameters
    window_length_min: int = 10  # minutes
    stride_min: int = 2  # minutes
    sampling_rate: float = 1.0  # Hz (samples per minute)
    
    # Data processing
    hr_range: Tuple[int, int] = (30, 220)
    spo2_range: Tuple[int, int] = (70, 100)
    resp_rate_range: Tuple[int, int] = (8, 40)
    
    # Model parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Architecture options
    model_type: str = "cnn"  # "cnn", "transformer", "multimodal"
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    
    # Evaluation
    stress_threshold: float = 0.5
    alignment_tolerance_min: int = 5  # minutes
    
    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        from pathlib import Path
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

@dataclass
class PreprocessingConfig(PipelineConfig):
    """Configuration for preprocessing-based pipeline"""
    
    # Preprocessing-specific parameters
    use_preprocessing: bool = True
    target_lengths: Dict[str, int] = field(default_factory=lambda: {
        'heart_rate': 3000,
        'stress': 3000,
        'oxygen_saturation': 2411,
        'respiratory_rate': 3000,
        'physical_activity': 3000,
        'physical_activity_calorie': 1895,
        'sleep': 403,
        'cgm': 2856
    })
    
    # Override sampling rate to match preprocessing (5-minute intervals)
    sampling_rate: float = 1.0 / 5.0  # 1 sample per 5 minutes = 0.2 Hz

# Global configuration
config = PipelineConfig()
