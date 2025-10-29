"""
Inference and Participant-Level Aggregation
===========================================

This module handles model inference and aggregation of predictions at the participant level.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class StressInference:
    """Handles model inference and participant-level aggregation"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict_windows(self, test_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Predict stress levels for all test windows
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary mapping participant_id to list of predictions
        """
        predictions_by_participant = defaultdict(list)
        
        with torch.no_grad():
            for batch in test_loader:
                # Move to device
                features = batch['features'].to(self.device)
                participant_ids = batch['participant_id']
                window_ids = batch['window_id']
                
                # Forward pass
                predictions = self.model(features)
                predictions = predictions.cpu().numpy().flatten()
                
                # Group by participant
                for i, participant_id in enumerate(participant_ids):
                    predictions_by_participant[participant_id].append({
                        'window_id': window_ids[i],
                        'prediction': predictions[i]
                    })
        
        # Convert to simple lists for easier processing
        result = {}
        for participant_id, preds in predictions_by_participant.items():
            result[participant_id] = [p['prediction'] for p in preds]
        
        return result
    
    def aggregate_participant_predictions(self, predictions_by_participant: Dict[str, List[float]], 
                                        stress_threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
        """
        Aggregate window-level predictions to participant-level metrics
        
        Args:
            predictions_by_participant: Window-level predictions by participant
            stress_threshold: Threshold for high-stress classification
            
        Returns:
            Dictionary of participant-level aggregated metrics
        """
        participant_metrics = {}
        
        for participant_id, predictions in predictions_by_participant.items():
            if not predictions:
                continue
            
            predictions_array = np.array(predictions)
            
            # Basic statistics
            mean_stress = np.mean(predictions_array)
            median_stress = np.median(predictions_array)
            std_stress = np.std(predictions_array)
            
            # High-stress percentage
            high_stress_pct = np.mean(predictions_array > stress_threshold) * 100
            
            # Volatility metrics
            volatility_std = std_stress
            volatility_mad = np.median(np.abs(predictions_array - median_stress))
            
            # Range
            min_stress = np.min(predictions_array)
            max_stress = np.max(predictions_array)
            stress_range = max_stress - min_stress
            
            participant_metrics[participant_id] = {
                'mean_stress': mean_stress,
                'median_stress': median_stress,
                'std_stress': std_stress,
                'high_stress_pct': high_stress_pct,
                'volatility_std': volatility_std,
                'volatility_mad': volatility_mad,
                'min_stress': min_stress,
                'max_stress': max_stress,
                'stress_range': stress_range,
                'num_windows': len(predictions_array)
            }
        
        return participant_metrics
    
    def compute_diurnal_patterns(self, predictions_by_participant: Dict[str, List[float]], 
                               window_times: Dict[str, List[str]]) -> Dict[str, Dict[int, float]]:
        """
        Compute diurnal stress patterns by hour of day
        
        Args:
            predictions_by_participant: Window-level predictions
            window_times: Window timestamps by participant
            
        Returns:
            Dictionary of hourly stress patterns by participant
        """
        diurnal_patterns = {}
        
        for participant_id, predictions in predictions_by_participant.items():
            if participant_id not in window_times:
                continue
            
            times = window_times[participant_id]
            hourly_stress = defaultdict(list)
            
            for i, (prediction, time_str) in enumerate(zip(predictions, times)):
                try:
                    # Parse timestamp and extract hour
                    timestamp = pd.to_datetime(time_str)
                    hour = timestamp.hour
                    hourly_stress[hour].append(prediction)
                except:
                    continue
            
            # Compute mean stress by hour
            hourly_means = {}
            for hour, stress_values in hourly_stress.items():
                if stress_values:
                    hourly_means[hour] = np.mean(stress_values)
            
            diurnal_patterns[participant_id] = hourly_means
        
        return diurnal_patterns
    
    def compute_weekday_weekend_patterns(self, predictions_by_participant: Dict[str, List[float]], 
                                       window_times: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Compute weekday vs weekend stress patterns
        
        Args:
            predictions_by_participant: Window-level predictions
            window_times: Window timestamps by participant
            
        Returns:
            Dictionary of weekday/weekend patterns by participant
        """
        weekday_patterns = {}
        
        for participant_id, predictions in predictions_by_participant.items():
            if participant_id not in window_times:
                continue
            
            times = window_times[participant_id]
            weekday_stress = []
            weekend_stress = []
            
            for prediction, time_str in zip(predictions, times):
                try:
                    timestamp = pd.to_datetime(time_str)
                    if timestamp.weekday() < 5:  # Monday = 0, Sunday = 6
                        weekday_stress.append(prediction)
                    else:
                        weekend_stress.append(prediction)
                except:
                    continue
            
            weekday_patterns[participant_id] = {
                'weekday_mean': np.mean(weekday_stress) if weekday_stress else np.nan,
                'weekend_mean': np.mean(weekend_stress) if weekend_stress else np.nan,
                'weekday_std': np.std(weekday_stress) if weekday_stress else np.nan,
                'weekend_std': np.std(weekend_stress) if weekend_stress else np.nan,
                'weekday_count': len(weekday_stress),
                'weekend_count': len(weekend_stress)
            }
        
        return weekday_patterns

class ParticipantLevelAnalyzer:
    """Analyzes participant-level aggregated predictions"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_participant_summary(self, participant_metrics: Dict[str, Dict[str, float]], 
                                 participants_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary DataFrame with participant-level metrics and metadata
        
        Args:
            participant_metrics: Aggregated participant metrics
            participants_df: Participant metadata
            
        Returns:
            Combined DataFrame with metrics and metadata
        """
        # Convert to DataFrame
        metrics_df = pd.DataFrame.from_dict(participant_metrics, orient='index')
        metrics_df.index.name = 'participant_id'
        metrics_df = metrics_df.reset_index()
        metrics_df['participant_id'] = metrics_df['participant_id'].astype(int)
        
        # Merge with participant metadata
        summary_df = metrics_df.merge(participants_df, on='participant_id', how='left')
        
        return summary_df
    
    def plot_participant_distributions(self, summary_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot distributions of participant-level metrics"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Mean stress by study group
        sns.boxplot(data=summary_df, x='study_group', y='mean_stress', ax=axes[0, 0])
        axes[0, 0].set_title('Mean Stress by Study Group')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # High stress percentage by study group
        sns.boxplot(data=summary_df, x='study_group', y='high_stress_pct', ax=axes[0, 1])
        axes[0, 1].set_title('High Stress Percentage by Study Group')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Volatility by study group
        sns.boxplot(data=summary_df, x='study_group', y='volatility_std', ax=axes[0, 2])
        axes[0, 2].set_title('Stress Volatility by Study Group')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Mean stress by age
        axes[1, 0].scatter(summary_df['age'], summary_df['mean_stress'])
        axes[1, 0].set_xlabel('Age')
        axes[1, 0].set_ylabel('Mean Stress')
        axes[1, 0].set_title('Mean Stress vs Age')
        
        # High stress percentage by age
        axes[1, 1].scatter(summary_df['age'], summary_df['high_stress_pct'])
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('High Stress Percentage')
        axes[1, 1].set_title('High Stress Percentage vs Age')
        
        # Volatility by age
        axes[1, 2].scatter(summary_df['age'], summary_df['volatility_std'])
        axes[1, 2].set_xlabel('Age')
        axes[1, 2].set_ylabel('Stress Volatility')
        axes[1, 2].set_title('Stress Volatility vs Age')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_diurnal_patterns(self, diurnal_patterns: Dict[str, Dict[int, float]], 
                            save_path: Optional[str] = None):
        """Plot diurnal stress patterns"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Individual participant patterns
        axes[0, 0].set_title('Individual Diurnal Patterns')
        for participant_id, hourly_means in list(diurnal_patterns.items())[:10]:  # Show first 10
            hours = list(hourly_means.keys())
            means = list(hourly_means.values())
            axes[0, 0].plot(hours, means, alpha=0.7, label=participant_id)
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Mean Stress')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Average pattern across all participants
        all_hours = set()
        for hourly_means in diurnal_patterns.values():
            all_hours.update(hourly_means.keys())
        
        avg_pattern = {}
        for hour in sorted(all_hours):
            hour_values = []
            for hourly_means in diurnal_patterns.values():
                if hour in hourly_means:
                    hour_values.append(hourly_means[hour])
            if hour_values:
                avg_pattern[hour] = np.mean(hour_values)
        
        axes[0, 1].plot(list(avg_pattern.keys()), list(avg_pattern.values()), 'o-')
        axes[0, 1].set_title('Average Diurnal Pattern')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Mean Stress')
        axes[0, 1].grid(True)
        
        # Heatmap of patterns
        pattern_matrix = []
        participant_ids = []
        for participant_id, hourly_means in diurnal_patterns.items():
            participant_ids.append(participant_id)
            row = []
            for hour in range(24):
                row.append(hourly_means.get(hour, np.nan))
            pattern_matrix.append(row)
        
        if pattern_matrix:
            im = axes[1, 0].imshow(pattern_matrix, aspect='auto', cmap='viridis')
            axes[1, 0].set_title('Diurnal Pattern Heatmap')
            axes[1, 0].set_xlabel('Hour of Day')
            axes[1, 0].set_ylabel('Participant')
            axes[1, 0].set_xticks(range(0, 24, 4))
            axes[1, 0].set_xticklabels(range(0, 24, 4))
            plt.colorbar(im, ax=axes[1, 0])
        
        # Distribution of peak stress hours
        peak_hours = []
        for hourly_means in diurnal_patterns.values():
            if hourly_means:
                peak_hour = max(hourly_means.keys(), key=lambda h: hourly_means[h])
                peak_hours.append(peak_hour)
        
        axes[1, 1].hist(peak_hours, bins=24, alpha=0.7)
        axes[1, 1].set_title('Distribution of Peak Stress Hours')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Number of Participants')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_weekday_weekend_patterns(self, weekday_patterns: Dict[str, Dict[str, float]], 
                                   save_path: Optional[str] = None):
        """Plot weekday vs weekend patterns"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract data
        weekday_means = []
        weekend_means = []
        participant_ids = []
        
        for participant_id, patterns in weekday_patterns.items():
            if not np.isnan(patterns['weekday_mean']) and not np.isnan(patterns['weekend_mean']):
                weekday_means.append(patterns['weekday_mean'])
                weekend_means.append(patterns['weekend_mean'])
                participant_ids.append(participant_id)
        
        # Scatter plot
        axes[0].scatter(weekday_means, weekend_means, alpha=0.7)
        axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Diagonal line
        axes[0].set_xlabel('Weekday Mean Stress')
        axes[0].set_ylabel('Weekend Mean Stress')
        axes[0].set_title('Weekday vs Weekend Stress')
        axes[0].grid(True)
        
        # Box plot
        data_for_box = []
        labels_for_box = []
        
        for participant_id, patterns in weekday_patterns.items():
            if not np.isnan(patterns['weekday_mean']):
                data_for_box.append(patterns['weekday_mean'])
                labels_for_box.append('Weekday')
            if not np.isnan(patterns['weekend_mean']):
                data_for_box.append(patterns['weekend_mean'])
                labels_for_box.append('Weekend')
        
        if data_for_box:
            axes[1].boxplot([data_for_box[i] for i, label in enumerate(labels_for_box) if label == 'Weekday'] + 
                           [data_for_box[i] for i, label in enumerate(labels_for_box) if label == 'Weekend'],
                           labels=['Weekday', 'Weekend'])
            axes[1].set_title('Weekday vs Weekend Stress Distribution')
            axes[1].set_ylabel('Mean Stress')
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, participant_metrics: Dict[str, Dict[str, float]], 
                    summary_df: pd.DataFrame, diurnal_patterns: Dict[str, Dict[int, float]], 
                    weekday_patterns: Dict[str, Dict[str, float]]):
        """Save all results to files"""
        
        # Save participant metrics
        metrics_df = pd.DataFrame.from_dict(participant_metrics, orient='index')
        metrics_df.to_csv(self.output_dir / "participant_metrics.csv")
        
        # Save summary
        summary_df.to_csv(self.output_dir / "participant_summary.csv", index=False)
        
        # Save diurnal patterns
        with open(self.output_dir / "diurnal_patterns.json", 'w') as f:
            json.dump(diurnal_patterns, f, indent=2)
        
        # Save weekday patterns
        with open(self.output_dir / "weekday_patterns.json", 'w') as f:
            json.dump(weekday_patterns, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
