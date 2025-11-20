"""
Feature Importance Analysis using SHAP and Permutation Importance
==================================================================

This script analyzes feature importance for the stress prediction model using:
1. SHAP (SHapley Additive exPlanations) - Best for deep learning models
2. Permutation Importance - Model-agnostic alternative

Features analyzed:
- 5 modalities: heart_rate, sleep, cgm, oxygen_saturation, respiratory_rate
- Each modality has 2 channels: value and mask
- Total: 10 features
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess"))

from config import PipelineConfig
from data_loader import EnhancedPreprocessingDataIngestion
from windowing import WindowingSystem
from data_splits import DataSplitter, StressDataset
from models import ModelFactory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Feature names based on windowing.py - MATCH TRAINING CONFIG
# Training uses: ['heart_rate', 'cgm', 'respiratory_rate'] + HR engineering
MODALITY_ORDER = ['heart_rate', 'cgm', 'respiratory_rate']  # Match training config
FEATURE_NAMES = []
for modality in MODALITY_ORDER:
    FEATURE_NAMES.extend([f"{modality}_value", f"{modality}_mask"])

# Add HR engineered features if enabled
HR_ENGINEERED_FEATURES = ['hr_mean', 'hr_std', 'hr_trend', 'hr_variability']
FEATURE_NAMES.extend(HR_ENGINEERED_FEATURES)

def apply_label_scaling(windows, scaler):
    """Apply label scaling to windows"""
    if scaler is None:
        return windows
    
    import copy
    scaled_windows = []
    for window in windows:
        new_window = copy.deepcopy(window)
        if new_window.get('stress_label') is not None:
            original_label = new_window['stress_label']
            scaled_label = scaler.transform([[original_label]])[0][0]
            new_window['stress_label'] = float(scaled_label)
        scaled_windows.append(new_window)
    
    return scaled_windows

def load_model_and_data(config: PipelineConfig, results_dir: Path, n_samples: int = 500):
    """Load model and prepare data for feature importance analysis"""
    
    model_path = results_dir / "models" / "best_model.pt"
    label_scaler_path = results_dir / "label_scaler.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    logger.info("=" * 80)
    logger.info("LOADING MODEL AND DATA FOR FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 80)
    
    # Load label scaler
    label_scaler = None
    if label_scaler_path.exists():
        with open(label_scaler_path, 'rb') as f:
            label_scaler = pickle.load(f)
        logger.info("✓ Loaded label scaler")
    
    # Set cache directory
    base_dir = Path(__file__).parent.parent
    config.cache_dir = str(base_dir / "data" / "processed" / "preprocessed_cache")
    
    # Load data
    logger.info("Loading data...")
    data_ingestion = EnhancedPreprocessingDataIngestion(config)
    windowing_system = WindowingSystem(config)
    data_splitter = DataSplitter(config)
    
    # Load windows (use test set for analysis)
    all_windows = []
    participants = data_ingestion.get_available_participants()
    
    logger.info(f"Processing participants for test set...")
    for i, participant_id in enumerate(participants):
        try:
            streams = data_ingestion.load_participant_streams(participant_id)
            if not streams:
                continue
            
            windows = windowing_system.create_windows(streams, participant_id)
            if windows:
                all_windows.extend(windows)
        except Exception as e:
            continue
    
    logger.info(f"Total windows: {len(all_windows)}")
    
    # Create splits
    participants_df = data_ingestion.participants_df
    train_windows, val_windows, test_windows = data_splitter.create_splits(
        all_windows, participants_df
    )
    
    logger.info(f"Train: {len(train_windows)}, Val: {len(val_windows)}, Test: {len(test_windows)}")
    
    # Apply label scaling
    if label_scaler is not None:
        test_windows = apply_label_scaling(test_windows, label_scaler)
        train_windows = apply_label_scaling(train_windows, label_scaler)
    
    # Create data loaders
    _, _, test_loader = data_splitter.create_dataloaders(
        train_windows, val_windows, test_windows
    )
    
    # Load model
    logger.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sample_batch = next(iter(test_loader))
    input_dim = sample_batch['features'].shape[2]
    
    model = ModelFactory.create_model(config.model_type, input_dim, config)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    logger.info("✓ Model loaded")
    
    # Prepare sample data for SHAP
    logger.info(f"Preparing {n_samples} samples for analysis...")
    X_samples = []
    y_samples = []
    
    sample_count = 0
    for batch in test_loader:
        if sample_count >= n_samples:
            break
        
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        
        batch_size = features.shape[0]
        remaining = n_samples - sample_count
        
        if remaining < batch_size:
            features = features[:remaining]
            labels = labels[:remaining]
        
        X_samples.append(features.cpu().numpy())
        y_samples.append(labels.cpu().numpy())
        sample_count += features.shape[0]
    
    X_samples = np.concatenate(X_samples, axis=0)
    y_samples = np.concatenate(y_samples, axis=0)
    
    logger.info(f"✓ Prepared {len(X_samples)} samples")
    logger.info(f"  Input shape: {X_samples.shape}")
    logger.info(f"  Label shape: {y_samples.shape}")
    
    return model, X_samples, y_samples, label_scaler, device, test_loader

def compute_permutation_importance(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_iterations: int = 5,
    n_samples: int = 200
) -> Dict[str, float]:
    """
    Compute permutation importance for each feature
    
    For time-series data, we permute each feature channel across all time steps
    """
    model.eval()
    
    # Use subset for faster computation
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
    else:
        X_subset = X
        y_subset = y
    
    # Baseline performance
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_subset).to(device)
        y_tensor = torch.FloatTensor(y_subset).to(device)
        baseline_pred = model(X_tensor)
        baseline_loss = nn.MSELoss()(baseline_pred, y_tensor).item()
    
    logger.info(f"Baseline MSE: {baseline_loss:.6f}")
    
    # Permutation importance for each feature
    feature_importance = {}
    
    # For each feature channel (10 total: 5 modalities × 2 channels)
    for feat_idx in tqdm(range(X.shape[2]), desc="Computing permutation importance"):
        feature_name = FEATURE_NAMES[feat_idx]
        importance_scores = []
        
        for iteration in range(n_iterations):
            # Create permuted copy
            X_permuted = X_subset.copy()
            
            # Permute this feature across all samples and time steps
            # Shape: [n_samples, time_steps, n_features]
            feature_values = X_permuted[:, :, feat_idx].flatten()
            np.random.shuffle(feature_values)
            X_permuted[:, :, feat_idx] = feature_values.reshape(X_permuted[:, :, feat_idx].shape)
            
            # Compute loss with permuted feature
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_permuted).to(device)
                permuted_pred = model(X_tensor)
                permuted_loss = nn.MSELoss()(permuted_pred, y_tensor).item()
            
            # Importance = increase in loss (higher is worse, so more important)
            importance = permuted_loss - baseline_loss
            importance_scores.append(importance)
        
        # Average importance across iterations
        avg_importance = np.mean(importance_scores)
        feature_importance[feature_name] = avg_importance
    
    return feature_importance

def compute_gradient_importance(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_samples: int = 200
) -> np.ndarray:
    """
    Compute feature importance using gradient-based method (Integrated Gradients approximation)
    This is faster than SHAP and doesn't require additional dependencies
    """
    model.eval()
    
    # Use subset for faster computation
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
    else:
        X_subset = X
        y_subset = y
    
    logger.info(f"Computing gradient-based importance on {len(X_subset)} samples...")
    
    # Convert to tensor and enable gradients
    X_tensor = torch.FloatTensor(X_subset).to(device)
    X_tensor.requires_grad_(True)
    y_tensor = torch.FloatTensor(y_subset).to(device)
    
    # Forward pass
    predictions = model(X_tensor)
    loss = nn.MSELoss()(predictions, y_tensor)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Get gradients w.r.t. input
    gradients = X_tensor.grad.data  # [batch, time, features]
    
    # Compute importance as mean absolute gradient across time and samples
    # Shape: [n_features]
    feature_importance = torch.mean(torch.abs(gradients), dim=(0, 1)).cpu().numpy()
    
    return feature_importance

def compute_shap_values(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    n_samples: int = 100,
    background_samples: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values using KernelExplainer
    
    Note: For time-series CNNs, KernelExplainer is used as it's more flexible
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Install with: pip install shap")
        logger.info("Using gradient-based importance instead...")
        raise ImportError("SHAP not available")
    
    model.eval()
    
    # Use subset for faster computation
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_explain = X[indices]
    else:
        X_explain = X
        n_samples = len(X)
    
    # Background samples for SHAP
    if len(X) > background_samples:
        bg_indices = np.random.choice(len(X), background_samples, replace=False)
        X_background = X[bg_indices]
    else:
        X_background = X
    
    logger.info(f"Using {len(X_explain)} samples for SHAP explanation")
    logger.info(f"Using {len(X_background)} background samples")
    
    # Wrapper function for model prediction
    def model_predict(X_flat):
        """
        Wrapper to handle flattened input from SHAP
        X_flat: [n_samples, time_steps * n_features] needs reshaping
        """
        # Reshape back to [batch, time, features]
        try:
            X_reshaped = X_flat.reshape(-1, X_explain.shape[1], X_explain.shape[2])
        except:
            # If reshape fails, pad or truncate
            total_features = X_explain.shape[1] * X_explain.shape[2]
            if X_flat.shape[1] != total_features:
                # Pad or truncate to match
                if X_flat.shape[1] < total_features:
                    padding = np.zeros((X_flat.shape[0], total_features - X_flat.shape[1]))
                    X_flat = np.concatenate([X_flat, padding], axis=1)
                else:
                    X_flat = X_flat[:, :total_features]
            X_reshaped = X_flat.reshape(-1, X_explain.shape[1], X_explain.shape[2])
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_reshaped).to(device)
            predictions = model(X_tensor)
            return predictions.cpu().numpy().flatten()
    
    # Flatten input for KernelExplainer
    # Shape: [n_samples, time_steps, n_features] -> [n_samples, time_steps * n_features]
    X_explain_flat = X_explain.reshape(len(X_explain), -1)
    X_background_flat = X_background.reshape(len(X_background), -1)
    
    logger.info("Computing SHAP values (this may take a while)...")
    
    # Use KernelExplainer with smaller background for speed
    explainer = shap.KernelExplainer(
        model_predict,
        X_background_flat[:min(30, len(X_background_flat))]  # Smaller background
    )
    
    # Compute SHAP values with limited samples for speed
    shap_values = explainer.shap_values(
        X_explain_flat[:min(50, len(X_explain_flat))],
        nsamples=50  # Reduced for speed
    )
    
    # Reshape SHAP values back to original shape
    n_explain = min(50, len(X_explain_flat))
    shap_values_reshaped = shap_values.reshape(
        n_explain,
        X_explain.shape[1],
        X_explain.shape[2]
    )
    
    # Average across time dimension to get feature-level importance
    # Shape: [n_samples, n_features]
    shap_values_feature = np.mean(np.abs(shap_values_reshaped), axis=1)
    
    # Average across samples to get global feature importance
    # Shape: [n_features]
    shap_importance = np.mean(shap_values_feature, axis=0)
    
    return shap_importance, shap_values_reshaped

def aggregate_modality_importance(feature_importance: Dict[str, float]) -> Dict[str, float]:
    """Aggregate feature importance by modality"""
    modality_importance = {}
    
    for modality in MODALITY_ORDER:
        value_key = f"{modality}_value"
        mask_key = f"{modality}_mask"
        
        value_imp = feature_importance.get(value_key, 0.0)
        mask_imp = feature_importance.get(mask_key, 0.0)
        
        # Combine value and mask importance
        modality_importance[modality] = value_imp + mask_imp
    
    # Add HR engineered features to heart_rate modality
    if 'heart_rate' in modality_importance:
        hr_engineered_imp = sum(
            feature_importance.get(feat, 0.0) 
            for feat in HR_ENGINEERED_FEATURES
        )
        modality_importance['heart_rate'] += hr_engineered_imp
    
    return modality_importance

def create_visualizations(
    permutation_importance: Dict[str, float],
    gradient_importance: np.ndarray,
    shap_importance: Optional[np.ndarray],
    modality_importance_perm: Dict[str, float],
    modality_importance_gradient: Dict[str, float],
    modality_importance_shap: Optional[Dict[str, float]],
    output_dir: Path
):
    """Create visualization plots for feature importance"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Feature-level permutation importance
    fig, ax = plt.subplots(figsize=(12, 8))
    features = list(permutation_importance.keys())
    importances = list(permutation_importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    features_sorted = [features[i] for i in sorted_idx]
    importances_sorted = [importances[i] for i in sorted_idx]
    
    bars = ax.barh(range(len(features_sorted)), importances_sorted)
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted)
    ax.set_xlabel('Permutation Importance (Increase in MSE)', fontsize=12)
    ax.set_title('Feature Importance: Permutation Method', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Color bars by modality
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODALITY_ORDER)))
    modality_colors = {}
    for i, modality in enumerate(MODALITY_ORDER):
        modality_colors[modality] = colors[i]
    
    for i, (bar, feature) in enumerate(zip(bars, features_sorted)):
        modality = feature.split('_')[0] if '_' in feature else feature
        for mod in MODALITY_ORDER:
            if modality.startswith(mod):
                bar.set_color(modality_colors[mod])
                break
    
    plt.tight_layout()
    plt.savefig(output_dir / 'permutation_importance_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: permutation_importance_features.png")
    
    # 2. Modality-level permutation importance
    fig, ax = plt.subplots(figsize=(10, 6))
    modalities = list(modality_importance_perm.keys())
    importances = list(modality_importance_perm.values())
    
    sorted_idx = np.argsort(importances)[::-1]
    modalities_sorted = [modalities[i] for i in sorted_idx]
    importances_sorted = [importances[i] for i in sorted_idx]
    
    bars = ax.barh(range(len(modalities_sorted)), importances_sorted, color=[modality_colors[m] for m in modalities_sorted])
    ax.set_yticks(range(len(modalities_sorted)))
    ax.set_yticklabels(modalities_sorted)
    ax.set_xlabel('Permutation Importance (Increase in MSE)', fontsize=12)
    ax.set_title('Modality Importance: Permutation Method', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'permutation_importance_modalities.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: permutation_importance_modalities.png")
    
    # 3. Gradient-based importance
    fig, ax = plt.subplots(figsize=(12, 8))
    
    gradient_dict = {FEATURE_NAMES[i]: gradient_importance[i] for i in range(len(FEATURE_NAMES))}
    features = list(gradient_dict.keys())
    importances = list(gradient_dict.values())
    
    sorted_idx = np.argsort(importances)[::-1]
    features_sorted = [features[i] for i in sorted_idx]
    importances_sorted = [importances[i] for i in sorted_idx]
    
    bars = ax.barh(range(len(features_sorted)), importances_sorted)
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted)
    ax.set_xlabel('Gradient Importance (Mean |Gradient|)', fontsize=12)
    ax.set_title('Feature Importance: Gradient-Based Method', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Color bars by modality
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODALITY_ORDER)))
    modality_colors = {}
    for i, modality in enumerate(MODALITY_ORDER):
        modality_colors[modality] = colors[i]
    
    for i, (bar, feature) in enumerate(zip(bars, features_sorted)):
        modality = feature.split('_')[0] if '_' in feature else feature
        for mod in MODALITY_ORDER:
            if modality.startswith(mod):
                bar.set_color(modality_colors[mod])
                break
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_importance_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: gradient_importance_features.png")
    
    # Modality-level gradient importance
    fig, ax = plt.subplots(figsize=(10, 6))
    modalities = list(modality_importance_gradient.keys())
    importances = list(modality_importance_gradient.values())
    
    sorted_idx = np.argsort(importances)[::-1]
    modalities_sorted = [modalities[i] for i in sorted_idx]
    importances_sorted = [importances[i] for i in sorted_idx]
    
    bars = ax.barh(range(len(modalities_sorted)), importances_sorted, color=[modality_colors[m] for m in modalities_sorted])
    ax.set_yticks(range(len(modalities_sorted)))
    ax.set_yticklabels(modalities_sorted)
    ax.set_xlabel('Gradient Importance (Mean |Gradient|)', fontsize=12)
    ax.set_title('Modality Importance: Gradient-Based Method', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_importance_modalities.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: gradient_importance_modalities.png")
    
    # 4. SHAP importance (if available)
    if shap_importance is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create feature importance dict from SHAP
        shap_dict = {FEATURE_NAMES[i]: shap_importance[i] for i in range(len(FEATURE_NAMES))}
        
        features = list(shap_dict.keys())
        importances = list(shap_dict.values())
        
        sorted_idx = np.argsort(importances)[::-1]
        features_sorted = [features[i] for i in sorted_idx]
        importances_sorted = [importances[i] for i in sorted_idx]
        
        bars = ax.barh(range(len(features_sorted)), importances_sorted)
        ax.set_yticks(range(len(features_sorted)))
        ax.set_yticklabels(features_sorted)
        ax.set_xlabel('SHAP Value (Mean |SHAP|)', fontsize=12)
        ax.set_title('Feature Importance: SHAP Method', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Color bars by modality
        for i, (bar, feature) in enumerate(zip(bars, features_sorted)):
            modality = feature.split('_')[0] if '_' in feature else feature
            for mod in MODALITY_ORDER:
                if modality.startswith(mod):
                    bar.set_color(modality_colors[mod])
                    break
        
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_importance_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: shap_importance_features.png")
        
        # Modality-level SHAP importance
        if modality_importance_shap is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            modalities = list(modality_importance_shap.keys())
            importances = list(modality_importance_shap.values())
            
            sorted_idx = np.argsort(importances)[::-1]
            modalities_sorted = [modalities[i] for i in sorted_idx]
            importances_sorted = [importances[i] for i in sorted_idx]
            
            bars = ax.barh(range(len(modalities_sorted)), importances_sorted, color=[modality_colors[m] for m in modalities_sorted])
            ax.set_yticks(range(len(modalities_sorted)))
            ax.set_yticklabels(modalities_sorted)
            ax.set_xlabel('SHAP Value (Mean |SHAP|)', fontsize=12)
            ax.set_title('Modality Importance: SHAP Method', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'shap_importance_modalities.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Saved: shap_importance_modalities.png")
    
    # 5. Comparison plot (all methods)
    n_methods = 2 if shap_importance is None else 3
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 6))
    if n_methods == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes
    
    # Permutation importance
    features = list(permutation_importance.keys())
    perm_imp = [permutation_importance[f] for f in features]
    sorted_idx = np.argsort(perm_imp)[::-1]
    features_sorted = [features[i] for i in sorted_idx]
    perm_imp_sorted = [perm_imp[i] for i in sorted_idx]
    
    ax1.barh(range(len(features_sorted)), perm_imp_sorted)
    ax1.set_yticks(range(len(features_sorted)))
    ax1.set_yticklabels(features_sorted)
    ax1.set_xlabel('Permutation Importance', fontsize=11)
    ax1.set_title('Permutation', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Gradient importance
    gradient_dict = {FEATURE_NAMES[i]: gradient_importance[i] for i in range(len(FEATURE_NAMES))}
    grad_imp = [gradient_dict[f] for f in features]
    grad_imp_sorted = [grad_imp[i] for i in sorted_idx]
    
    ax2.barh(range(len(features_sorted)), grad_imp_sorted)
    ax2.set_yticks(range(len(features_sorted)))
    ax2.set_yticklabels(features_sorted)
    ax2.set_xlabel('Gradient Importance', fontsize=11)
    ax2.set_title('Gradient-Based', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # SHAP importance (if available)
    if shap_importance is not None:
        shap_dict = {FEATURE_NAMES[i]: shap_importance[i] for i in range(len(FEATURE_NAMES))}
        shap_imp = [shap_dict[f] for f in features]
        shap_imp_sorted = [shap_imp[i] for i in sorted_idx]
        
        ax3.barh(range(len(features_sorted)), shap_imp_sorted)
        ax3.set_yticks(range(len(features_sorted)))
        ax3.set_yticklabels(features_sorted)
        ax3.set_xlabel('SHAP Value', fontsize=11)
        ax3.set_title('SHAP', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: feature_importance_comparison.png")

def save_results(
    permutation_importance: Dict[str, float],
    gradient_importance: np.ndarray,
    shap_importance: Optional[np.ndarray],
    modality_importance_perm: Dict[str, float],
    modality_importance_gradient: Dict[str, float],
    modality_importance_shap: Optional[Dict[str, float]],
    output_dir: Path
):
    """Save feature importance results to CSV and JSON"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Feature-level results
    results_df = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'permutation_importance': [permutation_importance.get(f, 0.0) for f in FEATURE_NAMES],
        'gradient_importance': gradient_importance.tolist(),
        'shap_importance': shap_importance.tolist() if shap_importance is not None else [None] * len(FEATURE_NAMES)
    })
    
    # Sort by permutation importance
    results_df = results_df.sort_values('permutation_importance', ascending=False)
    results_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    logger.info(f"✓ Saved: feature_importance.csv")
    
    # Modality-level results
    modality_df = pd.DataFrame({
        'modality': MODALITY_ORDER,
        'permutation_importance': [modality_importance_perm.get(m, 0.0) for m in MODALITY_ORDER],
        'gradient_importance': [modality_importance_gradient.get(m, 0.0) for m in MODALITY_ORDER],
        'shap_importance': [modality_importance_shap.get(m, 0.0) if modality_importance_shap else None for m in MODALITY_ORDER]
    })
    
    modality_df = modality_df.sort_values('permutation_importance', ascending=False)
    modality_df.to_csv(output_dir / 'modality_importance.csv', index=False)
    logger.info(f"✓ Saved: modality_importance.csv")
    
    # JSON summary
    summary = {
        'feature_importance': {
            'permutation': {f: float(v) for f, v in permutation_importance.items()},
            'gradient': {FEATURE_NAMES[i]: float(gradient_importance[i]) for i in range(len(FEATURE_NAMES))},
            'shap': {FEATURE_NAMES[i]: float(shap_importance[i]) for i in range(len(FEATURE_NAMES))} if shap_importance is not None else None
        },
        'modality_importance': {
            'permutation': {m: float(v) for m, v in modality_importance_perm.items()},
            'gradient': {m: float(v) for m, v in modality_importance_gradient.items()},
            'shap': {m: float(v) for m, v in modality_importance_shap.items()} if modality_importance_shap else None
        },
        'top_features_permutation': results_df.head(5)['feature'].tolist(),
        'top_modalities_permutation': modality_df.head(3)['modality'].tolist()
    }
    
    with open(output_dir / 'feature_importance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✓ Saved: feature_importance_summary.json")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE IMPORTANCE SUMMARY")
    logger.info("=" * 80)
    logger.info("\nTop 5 Features (Permutation Importance):")
    for i, row in results_df.head(5).iterrows():
        logger.info(f"  {i+1}. {row['feature']}: {row['permutation_importance']:.6f}")
    
    logger.info("\nTop 3 Modalities (Permutation Importance):")
    for i, row in modality_df.head(3).iterrows():
        logger.info(f"  {i+1}. {row['modality']}: {row['permutation_importance']:.6f}")

def main():
    """Main function"""
    base_dir = Path(__file__).parent.parent
    
    # Configuration - MUST MATCH TRAINING CONFIG
    config = PipelineConfig(
        data_root=str(base_dir / "AI-READI"),
        output_dir=str(base_dir / "data" / "processed" / "results_improved_v2"),
        model_type="cnn",
        window_length_min=60,
        stride_min=60,
        batch_size=64,
        num_epochs=300,
        learning_rate=0.001,
        hidden_dim=256,
        num_layers=6,
        dropout=0.3,
        early_stopping_patience=30
    )
    
    # Set feature selection and HR engineering to match training
    config.selected_modalities = ['heart_rate', 'cgm', 'respiratory_rate']
    config.enable_heart_rate_engineering = True
    
    results_dir = Path(config.output_dir)
    output_dir = results_dir / "feature_importance"
    
    try:
        # Load model and data
        model, X_samples, y_samples, label_scaler, device, test_loader = load_model_and_data(
            config, results_dir, n_samples=500
        )
        
        # 1. Compute Permutation Importance
        logger.info("\n" + "=" * 80)
        logger.info("COMPUTING PERMUTATION IMPORTANCE")
        logger.info("=" * 80)
        permutation_importance = compute_permutation_importance(
            model, X_samples, y_samples, device, n_iterations=5, n_samples=200
        )
        
        # Aggregate by modality
        modality_importance_perm = aggregate_modality_importance(permutation_importance)
        
        # 2. Compute Gradient-based Importance (fast, no extra dependencies)
        logger.info("\n" + "=" * 80)
        logger.info("COMPUTING GRADIENT-BASED IMPORTANCE")
        logger.info("=" * 80)
        gradient_importance = compute_gradient_importance(
            model, X_samples, y_samples, device, n_samples=200
        )
        
        # Aggregate gradient importance by modality
        gradient_dict = {FEATURE_NAMES[i]: gradient_importance[i] for i in range(len(FEATURE_NAMES))}
        modality_importance_gradient = {}
        for modality in MODALITY_ORDER:
            value_key = f"{modality}_value"
            mask_key = f"{modality}_mask"
            modality_importance_gradient[modality] = gradient_dict.get(value_key, 0.0) + gradient_dict.get(mask_key, 0.0)
        
        # Add HR engineered features to heart_rate modality
        if 'heart_rate' in modality_importance_gradient:
            hr_engineered_imp = sum(
                gradient_dict.get(feat, 0.0) 
                for feat in HR_ENGINEERED_FEATURES
            )
            modality_importance_gradient['heart_rate'] += hr_engineered_imp
        
        logger.info("✓ Gradient-based importance computed successfully")
        
        # 3. Compute SHAP values (optional, can be slow, requires SHAP package)
        logger.info("\n" + "=" * 80)
        logger.info("COMPUTING SHAP VALUES (OPTIONAL)")
        logger.info("=" * 80)
        shap_importance = None
        modality_importance_shap = None
        
        try:
            shap_importance, shap_values = compute_shap_values(
                model, X_samples, device, n_samples=50, background_samples=30
            )
            
            # Aggregate SHAP by modality
            shap_dict = {FEATURE_NAMES[i]: shap_importance[i] for i in range(len(FEATURE_NAMES))}
            modality_importance_shap = {}
            for modality in MODALITY_ORDER:
                value_key = f"{modality}_value"
                mask_key = f"{modality}_mask"
                modality_importance_shap[modality] = shap_dict.get(value_key, 0.0) + shap_dict.get(mask_key, 0.0)
            
            # Add HR engineered features to heart_rate modality
            if 'heart_rate' in modality_importance_shap:
                hr_engineered_imp = sum(
                    shap_dict.get(feat, 0.0) 
                    for feat in HR_ENGINEERED_FEATURES
                )
                modality_importance_shap['heart_rate'] += hr_engineered_imp
            
            logger.info("✓ SHAP values computed successfully")
        except ImportError:
            logger.info("SHAP not installed. Skipping SHAP analysis.")
            logger.info("Install with: pip install shap")
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            logger.info("Continuing with permutation and gradient importance...")
        
        # 3. Create visualizations
        logger.info("\n" + "=" * 80)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("=" * 80)
        create_visualizations(
            permutation_importance,
            gradient_importance,
            shap_importance,
            modality_importance_perm,
            modality_importance_gradient,
            modality_importance_shap,
            output_dir
        )
        
        # 4. Save results
        logger.info("\n" + "=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)
        save_results(
            permutation_importance,
            gradient_importance,
            shap_importance,
            modality_importance_perm,
            modality_importance_gradient,
            modality_importance_shap,
            output_dir
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE IMPORTANCE ANALYSIS COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

