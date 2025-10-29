"""
Comprehensive Evaluation Metrics
===============================

This module provides comprehensive evaluation metrics for stress prediction models,
including window-level, participant-level, temporal, and subgroup analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class StressEvaluator:
    """Comprehensive evaluator for stress prediction models"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_window_level(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate model performance at window level
        
        Args:
            y_true: True stress values
            y_pred: Predicted stress values
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary of window-level metrics
        """
        # Regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Correlation metrics
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        
        # Classification metrics (using threshold)
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        f1 = f1_score(y_true_binary, y_pred_binary, average='macro')
        precision = precision_score(y_true_binary, y_pred_binary, average='macro')
        recall = recall_score(y_true_binary, y_pred_binary, average='macro')
        
        # AUC (treating as binary classification)
        try:
            auc = roc_auc_score(y_true_binary, y_pred_binary)
        except ValueError:
            auc = np.nan
        
        # Calibration metrics
        calibration_error = self._compute_calibration_error(y_true, y_pred)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'f1_macro': f1,
            'precision_macro': precision,
            'recall_macro': recall,
            'auc': auc,
            'calibration_error': calibration_error
        }
    
    def evaluate_participant_level(self, participant_metrics: Dict[str, Dict[str, float]], 
                                  participants_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance at participant level
        
        Args:
            participant_metrics: Aggregated participant metrics
            participants_df: Participant metadata
            
        Returns:
            Dictionary of participant-level evaluation metrics
        """
        # Create summary DataFrame
        metrics_df = pd.DataFrame.from_dict(participant_metrics, orient='index')
        metrics_df.index.name = 'participant_id'
        metrics_df = metrics_df.reset_index()
        metrics_df['participant_id'] = metrics_df['participant_id'].astype(int)
        
        summary_df = metrics_df.merge(participants_df, on='participant_id', how='left')
        
        # Overall statistics
        overall_stats = {
            'mean_stress_mean': summary_df['mean_stress'].mean(),
            'mean_stress_std': summary_df['mean_stress'].std(),
            'high_stress_pct_mean': summary_df['high_stress_pct'].mean(),
            'high_stress_pct_std': summary_df['high_stress_pct'].std(),
            'volatility_mean': summary_df['volatility_std'].mean(),
            'volatility_std': summary_df['volatility_std'].std()
        }
        
        # Group-wise statistics
        group_stats = {}
        for group in summary_df['study_group'].unique():
            group_data = summary_df[summary_df['study_group'] == group]
            group_stats[group] = {
                'count': len(group_data),
                'mean_stress_mean': group_data['mean_stress'].mean(),
                'mean_stress_std': group_data['mean_stress'].std(),
                'high_stress_pct_mean': group_data['high_stress_pct'].mean(),
                'high_stress_pct_std': group_data['high_stress_pct'].std(),
                'volatility_mean': group_data['volatility_std'].mean(),
                'volatility_std': group_data['volatility_std'].std()
            }
        
        # Age correlations
        age_correlations = {}
        for metric in ['mean_stress', 'high_stress_pct', 'volatility_std']:
            valid_data = summary_df.dropna(subset=[metric, 'age'])
            if len(valid_data) > 1:
                corr, p_val = pearsonr(valid_data['age'], valid_data[metric])
                age_correlations[f'{metric}_age_correlation'] = corr
                age_correlations[f'{metric}_age_p_value'] = p_val
        
        return {
            'overall': overall_stats,
            'by_group': group_stats,
            'age_correlations': age_correlations
        }
    
    def evaluate_temporal_consistency(self, predictions_by_participant: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Evaluate temporal consistency of predictions
        
        Args:
            predictions_by_participant: Window-level predictions by participant
            
        Returns:
            Dictionary of temporal consistency metrics
        """
        temporal_metrics = {}
        
        all_autocorrelations = []
        all_serial_correlations = []
        
        for participant_id, predictions in predictions_by_participant.items():
            if len(predictions) < 2:
                continue
            
            pred_array = np.array(predictions)
            
            # Serial correlation (lag-1 autocorrelation)
            if len(pred_array) > 1:
                serial_corr = np.corrcoef(pred_array[:-1], pred_array[1:])[0, 1]
                if not np.isnan(serial_corr):
                    all_serial_correlations.append(serial_corr)
            
            # Autocorrelation at different lags
            for lag in range(1, min(10, len(pred_array))):
                if len(pred_array) > lag:
                    autocorr = np.corrcoef(pred_array[:-lag], pred_array[lag:])[0, 1]
                    if not np.isnan(autocorr):
                        all_autocorrelations.append(autocorr)
        
        temporal_metrics['mean_serial_correlation'] = np.mean(all_serial_correlations) if all_serial_correlations else np.nan
        temporal_metrics['std_serial_correlation'] = np.std(all_serial_correlations) if all_serial_correlations else np.nan
        temporal_metrics['mean_autocorrelation'] = np.mean(all_autocorrelations) if all_autocorrelations else np.nan
        temporal_metrics['std_autocorrelation'] = np.std(all_autocorrelations) if all_autocorrelations else np.nan
        
        return temporal_metrics
    
    def evaluate_subgroup_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    participant_ids: List[str], participants_df: pd.DataFrame,
                                    threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance across different subgroups
        
        Args:
            y_true: True stress values
            y_pred: Predicted stress values
            participant_ids: Participant IDs for each prediction
            participants_df: Participant metadata
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary of subgroup performance metrics
        """
        # Create DataFrame with predictions and metadata
        pred_df = pd.DataFrame({
            'participant_id': participant_ids,
            'y_true': y_true,
            'y_pred': y_pred
        })
        
        # Convert participant_id to int for merging
        pred_df['participant_id'] = pred_df['participant_id'].astype(int)
        
        # Merge with participant metadata
        pred_df = pred_df.merge(participants_df, on='participant_id', how='left')
        
        subgroup_metrics = {}
        
        # By study group
        for group in pred_df['study_group'].unique():
            group_data = pred_df[pred_df['study_group'] == group]
            if len(group_data) > 0:
                subgroup_metrics[f'study_group_{group}'] = self.evaluate_window_level(
                    group_data['y_true'].values, group_data['y_pred'].values, threshold
                )
        
        # By clinical site
        for site in pred_df['clinical_site'].unique():
            site_data = pred_df[pred_df['clinical_site'] == site]
            if len(site_data) > 0:
                subgroup_metrics[f'clinical_site_{site}'] = self.evaluate_window_level(
                    site_data['y_true'].values, site_data['y_pred'].values, threshold
                )
        
        # By age groups
        pred_df['age_group'] = pd.cut(pred_df['age'], bins=[0, 50, 65, 100], labels=['young', 'middle', 'old'])
        for age_group in pred_df['age_group'].unique():
            age_data = pred_df[pred_df['age_group'] == age_group]
            if len(age_data) > 0:
                subgroup_metrics[f'age_group_{age_group}'] = self.evaluate_window_level(
                    age_data['y_true'].values, age_data['y_pred'].values, threshold
                )
        
        return subgroup_metrics
    
    def _compute_calibration_error(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)"""
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Compute accuracy in this bin
                accuracy_in_bin = (y_true[in_bin] >= 0.5).mean()
                avg_confidence_in_bin = y_pred[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def plot_evaluation_results(self, window_metrics: Dict[str, float], 
                              participant_metrics: Dict[str, Dict[str, float]],
                              subgroup_metrics: Dict[str, Dict[str, float]],
                              save_path: Optional[str] = None):
        """Plot comprehensive evaluation results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Window-level metrics
        metrics_names = ['mae', 'rmse', 'r2', 'f1_macro', 'auc', 'calibration_error']
        metrics_values = [window_metrics.get(name, np.nan) for name in metrics_names]
        
        axes[0, 0].bar(metrics_names, metrics_values)
        axes[0, 0].set_title('Window-Level Metrics')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Participant-level mean stress by group
        group_names = list(participant_metrics['by_group'].keys())
        group_means = [participant_metrics['by_group'][group]['mean_stress_mean'] for group in group_names]
        group_stds = [participant_metrics['by_group'][group]['mean_stress_std'] for group in group_names]
        
        axes[0, 1].bar(group_names, group_means, yerr=group_stds, capsize=5)
        axes[0, 1].set_title('Mean Stress by Study Group')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # High stress percentage by group
        group_high_stress = [participant_metrics['by_group'][group]['high_stress_pct_mean'] for group in group_names]
        group_high_stress_std = [participant_metrics['by_group'][group]['high_stress_pct_std'] for group in group_names]
        
        axes[0, 2].bar(group_names, group_high_stress, yerr=group_high_stress_std, capsize=5)
        axes[0, 2].set_title('High Stress Percentage by Study Group')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Subgroup MAE comparison
        subgroup_names = list(subgroup_metrics.keys())
        subgroup_mae = [subgroup_metrics[name]['mae'] for name in subgroup_names]
        
        axes[1, 0].bar(subgroup_names, subgroup_mae)
        axes[1, 0].set_title('MAE by Subgroup')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Subgroup F1 comparison
        subgroup_f1 = [subgroup_metrics[name]['f1_macro'] for name in subgroup_names]
        
        axes[1, 1].bar(subgroup_names, subgroup_f1)
        axes[1, 1].set_title('F1 Score by Subgroup')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Subgroup AUC comparison
        subgroup_auc = [subgroup_metrics[name]['auc'] for name in subgroup_names]
        
        axes[1, 2].bar(subgroup_names, subgroup_auc)
        axes[1, 2].set_title('AUC by Subgroup')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            threshold: float = 0.5, save_path: Optional[str] = None):
        """Plot confusion matrix for binary classification"""
        
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Threshold = {threshold})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             n_bins: int = 10, save_path: Optional[str] = None):
        """Plot calibration curve"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
            prop_in_bin = in_bin.sum()
            
            if prop_in_bin > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append((y_true[in_bin] >= 0.5).mean())
                bin_counts.append(prop_in_bin)
        
        plt.figure(figsize=(8, 6))
        plt.plot(bin_centers, bin_accuracies, 'o-', label='Model')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_evaluation_results(self, window_metrics: Dict[str, float],
                              participant_metrics: Dict[str, Dict[str, float]],
                              temporal_metrics: Dict[str, float],
                              subgroup_metrics: Dict[str, Dict[str, float]]):
        """Save all evaluation results to files"""
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        # Convert all metrics to JSON-serializable format
        window_metrics = convert_numpy_types(window_metrics)
        participant_metrics = convert_numpy_types(participant_metrics)
        temporal_metrics = convert_numpy_types(temporal_metrics)
        subgroup_metrics = convert_numpy_types(subgroup_metrics)
        
        # Save window-level metrics
        with open(self.output_dir / "window_level_metrics.json", 'w') as f:
            json.dump(window_metrics, f, indent=2)
        
        # Save participant-level metrics
        with open(self.output_dir / "participant_level_metrics.json", 'w') as f:
            json.dump(participant_metrics, f, indent=2)
        
        # Save temporal metrics
        with open(self.output_dir / "temporal_metrics.json", 'w') as f:
            json.dump(temporal_metrics, f, indent=2)
        
        # Save subgroup metrics
        with open(self.output_dir / "subgroup_metrics.json", 'w') as f:
            json.dump(subgroup_metrics, f, indent=2)
        
        logger.info(f"Evaluation results saved to {self.output_dir}")
