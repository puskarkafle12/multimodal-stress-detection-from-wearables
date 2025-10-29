"""
Windowing and Label Alignment Module
===================================

This module handles sliding window creation and label alignment for time series data.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class WindowingSystem:
    """Handles windowing and label alignment for multimodal time series"""
    
    def __init__(self, config):
        self.config = config
        self.window_length_min = config.window_length_min
        self.stride_min = config.stride_min
        self.sampling_rate = config.sampling_rate
        self.alignment_tolerance_min = config.alignment_tolerance_min
        
        # Calculate window parameters
        self.window_length_samples = int(self.window_length_min * self.sampling_rate * 60)  # samples
        self.stride_samples = int(self.stride_min * self.sampling_rate * 60)  # samples
        
    def create_windows(self, streams: Dict[str, pd.DataFrame], participant_id: str) -> List[Dict]:
        """
        Create sliding windows from synchronized streams
        
        Args:
            streams: Dictionary of synchronized modality streams
            participant_id: Participant identifier
            
        Returns:
            List of window dictionaries with features and labels
        """
        if not streams:
            return []
        
        # Get common timestamps (assuming all streams are synchronized)
        first_stream = list(streams.values())[0]
        timestamps = first_stream['timestamp'].values
        
        windows = []
        
        # Create sliding windows
        for start_idx in range(0, len(timestamps) - self.window_length_samples + 1, self.stride_samples):
            end_idx = start_idx + self.window_length_samples
            
            # Extract window data
            window_start_time = timestamps[start_idx]
            window_end_time = timestamps[end_idx - 1]
            window_center_time = timestamps[start_idx + self.window_length_samples // 2]
            
            # Ensure timezone-naive timestamps for consistency
            if hasattr(window_center_time, 'tz') and window_center_time.tz is not None:
                # Convert to timezone-naive
                window_center_time = window_center_time.tz_localize(None)
            else:
                # Already timezone-naive
                pass
            
            # Build feature tensor for this window
            window_features = self._build_window_features(streams, start_idx, end_idx)
            
            if window_features is None:
                continue
            
            # Align stress label
            stress_label = self._align_stress_label(streams, window_center_time)
            
            if stress_label is not None:
                window_data = {
                    'participant_id': participant_id,
                    'window_id': f"{participant_id}_{start_idx}",
                    'start_time': window_start_time,
                    'end_time': window_end_time,
                    'center_time': window_center_time,
                    'features': window_features,
                    'stress_label': stress_label
                }
                windows.append(window_data)
        
        logger.info(f"Created {len(windows)} windows for participant {participant_id}")
        return windows
    
    def _build_window_features(self, streams: Dict[str, pd.DataFrame], start_idx: int, end_idx: int) -> Optional[np.ndarray]:
        """
        Build feature tensor for a window
        
        Args:
            streams: Synchronized modality streams
            start_idx: Start index for window
            end_idx: End index for window
            
        Returns:
            Feature tensor of shape [T, F] where T=window_length_samples, F=num_features
        """
        features = []
        feature_names = []
        
        # Define modality order for consistent feature ordering
        modality_order = [
            'heart_rate', 'oxygen_saturation', 'respiratory_rate', 
            'physical_activity', 'physical_activity_calorie', 'sleep', 'cgm'
        ]
        
        for modality in modality_order:
            if modality not in streams:
                # Add zeros for missing modalities
                features.append(np.zeros((self.window_length_samples, 2)))  # value + mask
                feature_names.extend([f"{modality}_value", f"{modality}_mask"])
                continue
            
            df = streams[modality]
            
            # Extract window data
            window_values = df['value'].iloc[start_idx:end_idx].values
            window_masks = df['mask'].iloc[start_idx:end_idx].values
            
            # Handle NaN values
            window_values = np.nan_to_num(window_values, nan=0.0)
            
            # Stack value and mask channels
            modality_features = np.stack([window_values, window_masks], axis=1)
            features.append(modality_features)
            feature_names.extend([f"{modality}_value", f"{modality}_mask"])
        
        if not features:
            return None
        
        # Concatenate all modality features
        feature_tensor = np.concatenate(features, axis=1)  # [T, F]
        
        return feature_tensor
    
    def _align_stress_label(self, streams: Dict[str, pd.DataFrame], window_center_time: pd.Timestamp) -> Optional[float]:
        """
        Align stress label to window center time
        
        Args:
            streams: Synchronized modality streams
            window_center_time: Center timestamp of the window
            
        Returns:
            Stress label value or None if no valid label found
        """
        if 'stress' not in streams:
            return None
        
        stress_df = streams['stress']
        
        # Find closest stress measurement within tolerance
        time_diffs = np.abs((stress_df['timestamp'] - window_center_time).dt.total_seconds() / 60)
        closest_idx = np.argmin(time_diffs)
        
        if time_diffs.iloc[closest_idx] <= self.alignment_tolerance_min:
            # Check if the measurement is valid (mask = 1)
            if stress_df['mask'].iloc[closest_idx] == 1:
                return stress_df['value'].iloc[closest_idx]
        
        return None
    
    def create_proxy_stress_labels(self, streams: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create proxy stress labels based on physiological indicators
        
        This is an alternative to EMA stress labels when they're not available
        """
        if not streams:
            return pd.DataFrame()
        
        # Get common timestamps
        first_stream = list(streams.values())[0]
        timestamps = first_stream['timestamp'].values
        
        proxy_labels = []
        
        for i, timestamp in enumerate(timestamps):
            # Extract features for this time point
            features = {}
            
            for modality, df in streams.items():
                if modality == 'stress':  # Skip actual stress labels
                    continue
                    
                if i < len(df):
                    features[f"{modality}_value"] = df['value'].iloc[i]
                    features[f"{modality}_mask"] = df['mask'].iloc[i]
                else:
                    features[f"{modality}_value"] = 0.0
                    features[f"{modality}_mask"] = 0
            
            # Compute proxy stress score
            stress_score = self._compute_proxy_stress_score(features)
            
            proxy_labels.append({
                'timestamp': timestamp,
                'proxy_stress': stress_score,
                'mask': 1 if not np.isnan(stress_score) else 0
            })
        
        return pd.DataFrame(proxy_labels)
    
    def _compute_proxy_stress_score(self, features: Dict[str, float]) -> float:
        """
        Compute proxy stress score from physiological features
        
        This is a simple heuristic - in practice, you might want to use
        more sophisticated methods like HRV analysis
        """
        stress_score = 0.0
        weight_sum = 0.0
        
        # Heart rate contribution (higher HR = higher stress)
        if features.get('heart_rate_mask', 0) == 1:
            hr_value = features['heart_rate_value']
            if 60 <= hr_value <= 100:  # Normal range
                hr_stress = 0.2
            elif 100 < hr_value <= 120:  # Elevated
                hr_stress = 0.5
            elif hr_value > 120:  # High
                hr_stress = 0.8
            else:  # Low (could indicate stress too)
                hr_stress = 0.3
            
            stress_score += hr_stress * 0.4  # Weight heart rate heavily
            weight_sum += 0.4
        
        # Respiratory rate contribution
        if features.get('respiratory_rate_mask', 0) == 1:
            rr_value = features['respiratory_rate_value']
            if 12 <= rr_value <= 20:  # Normal range
                rr_stress = 0.2
            elif rr_value > 20:  # Elevated
                rr_stress = 0.6
            else:  # Low
                rr_stress = 0.4
            
            stress_score += rr_stress * 0.3
            weight_sum += 0.3
        
        # Physical activity contribution (lower activity = higher stress)
        if features.get('physical_activity_mask', 0) == 1:
            pa_value = features['physical_activity_value']
            if pa_value == 0:  # Sedentary
                pa_stress = 0.6
            elif pa_value == 1:  # Light activity
                pa_stress = 0.3
            elif pa_value >= 2:  # Moderate/vigorous
                pa_stress = 0.1
            
            stress_score += pa_stress * 0.2
            weight_sum += 0.2
        
        # Sleep stage contribution (awake = higher stress)
        if features.get('sleep_mask', 0) == 1:
            sleep_value = features['sleep_value']
            if sleep_value == 0:  # Awake
                sleep_stress = 0.7
            elif sleep_value == 1:  # Light sleep
                sleep_stress = 0.3
            else:  # Deep/REM sleep
                sleep_stress = 0.1
            
            stress_score += sleep_stress * 0.1
            weight_sum += 0.1
        
        # Normalize by weight sum
        if weight_sum > 0:
            stress_score = stress_score / weight_sum
        else:
            stress_score = np.nan
        
        return stress_score
    
    def get_window_statistics(self, windows: List[Dict]) -> Dict[str, float]:
        """Get statistics about created windows"""
        if not windows:
            return {}
        
        window_lengths = [len(w['features']) for w in windows]
        stress_labels = [w['stress_label'] for w in windows if w['stress_label'] is not None]
        
        stats = {
            'total_windows': len(windows),
            'windows_with_labels': len(stress_labels),
            'label_coverage': len(stress_labels) / len(windows) if windows else 0,
            'avg_window_length': np.mean(window_lengths),
            'stress_label_mean': np.mean(stress_labels) if stress_labels else np.nan,
            'stress_label_std': np.std(stress_labels) if stress_labels else np.nan,
            'stress_label_min': np.min(stress_labels) if stress_labels else np.nan,
            'stress_label_max': np.max(stress_labels) if stress_labels else np.nan
        }
        
        return stats
