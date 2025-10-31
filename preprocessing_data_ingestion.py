"""
Preprocessing-Based Data Ingestion Module
=======================================

This module integrates the preprocessing methods from the preprocess folder
into the main pipeline.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime

# Import preprocessing methods
import sys
sys.path.append('preprocess')
from wearable_loader import (
    load_heartrate, load_stress, load_oxygen, load_resp, 
    load_activity, load_calories, load_sleep
)
from cgm_loader import load_cgm

logger = logging.getLogger(__name__)

class PreprocessingDataIngestion:
    """Handles data ingestion using preprocessing methods from preprocess folder"""
    
    def __init__(self, config):
        self.config = config
        self.data_root = config.data_root
        self.sampling_rate = config.sampling_rate
        
        # Define modality paths (same as original)
        self.modality_paths = {
            'heart_rate': f"{self.data_root}/wearable_activity_monitor/heart_rate/garmin_vivosmart5",
            'stress': f"{self.data_root}/wearable_activity_monitor/stress/garmin_vivosmart5", 
            'oxygen_saturation': f"{self.data_root}/wearable_activity_monitor/oxygen_saturation/garmin_vivosmart5",
            'respiratory_rate': f"{self.data_root}/wearable_activity_monitor/respiratory_rate/garmin_vivosmart5",
            'physical_activity': f"{self.data_root}/wearable_activity_monitor/physical_activity/garmin_vivosmart5",
            'physical_activity_calorie': f"{self.data_root}/wearable_activity_monitor/physical_activity_calorie/garmin_vivosmart5",
            'sleep': f"{self.data_root}/wearable_activity_monitor/sleep/garmin_vivosmart5",
            'cgm': f"{self.data_root}/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6"
        }
        
        # Load participant metadata
        self.participants_df = pd.read_csv(f"{self.data_root}/participants.tsv", sep='\t')
        
        # Define target lengths for each modality (from preprocess folder)
        self.target_lengths = {
            'heart_rate': 3000,
            'stress': 3000,
            'oxygen_saturation': 2411,
            'respiratory_rate': 3000,
            'physical_activity': 3000,
            'physical_activity_calorie': 1895,
            'sleep': 403,
            'cgm': 2856
        }
        
    def get_available_participants(self) -> List[str]:
        """Get list of available participants with actual data files"""
        # Check which participants actually have data files
        available_participants = []
        
        # Check heart rate data as a proxy for available participants
        heart_rate_path = self.modality_paths['heart_rate']
        if os.path.exists(heart_rate_path):
            participant_dirs = [d for d in os.listdir(heart_rate_path) if os.path.isdir(os.path.join(heart_rate_path, d))]
            available_participants = sorted(participant_dirs)
        
        return available_participants
        
    def load_participant_streams(self, participant_id: str) -> Dict[str, pd.DataFrame]:
        """
        Load all available data streams for a participant using preprocessing methods
        
        Args:
            participant_id: Participant identifier (e.g., "1023")
            
        Returns:
            Dictionary mapping modality names to DataFrames with columns [timestamp, value, mask]
        """
        streams = {}
        
        # Define file paths for this participant
        file_paths = {
            'heart_rate': f"{self.modality_paths['heart_rate']}/{participant_id}/{participant_id}_heartrate.json",
            'stress': f"{self.modality_paths['stress']}/{participant_id}/{participant_id}_stress.json",
            'oxygen_saturation': f"{self.modality_paths['oxygen_saturation']}/{participant_id}/{participant_id}_oxygensaturation.json",
            'respiratory_rate': f"{self.modality_paths['respiratory_rate']}/{participant_id}/{participant_id}_respiratoryrate.json",
            'physical_activity': f"{self.modality_paths['physical_activity']}/{participant_id}/{participant_id}_activity.json",
            'physical_activity_calorie': f"{self.modality_paths['physical_activity_calorie']}/{participant_id}/{participant_id}_calorie.json",
            'sleep': f"{self.modality_paths['sleep']}/{participant_id}/{participant_id}_sleep.json",
            'cgm': f"{self.modality_paths['cgm']}/{participant_id}/{participant_id}_DEX.json"
        }
        
        # Load each modality using preprocessing methods
        for modality, file_path in file_paths.items():
            try:
                if os.path.exists(file_path):
                    # Use preprocessing methods to load and process data
                    values, mask = self._load_with_preprocessing(modality, file_path)
                    
                    if values is not None and mask is not None:
                        # Create DataFrame with timestamp, value, mask columns
                        stream_data = self._create_stream_dataframe(values, mask, modality)
                        if stream_data is not None:
                            streams[modality] = stream_data
                            logger.info(f"Loaded {modality} for participant {participant_id}: {len(stream_data)} samples")
                            
            except Exception as e:
                logger.warning(f"Failed to load {modality} for participant {participant_id}: {e}")
                
        return streams
    
    def _load_with_preprocessing(self, modality: str, file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load data using preprocessing methods"""
        
        target_len = self.target_lengths.get(modality, 3000)
        
        try:
            if modality == 'heart_rate':
                return load_heartrate(file_path, target_len)
            elif modality == 'stress':
                return load_stress(file_path, target_len)
            elif modality == 'oxygen_saturation':
                return load_oxygen(file_path, target_len)
            elif modality == 'respiratory_rate':
                return load_resp(file_path, target_len)
            elif modality == 'physical_activity':
                return load_activity(file_path, target_len)
            elif modality == 'physical_activity_calorie':
                return load_calories(file_path, target_len)
            elif modality == 'sleep':
                return load_sleep(file_path, target_len)
            elif modality == 'cgm':
                return load_cgm(file_path, target_len)
            else:
                return None, None
                
        except Exception as e:
            logger.warning(f"Error loading {modality}: {e}")
            return None, None
    
    def _create_stream_dataframe(self, values: np.ndarray, mask: np.ndarray, modality: str) -> Optional[pd.DataFrame]:
        """Create DataFrame from preprocessed values and mask"""
        
        if values is None or mask is None or len(values) == 0:
            return None
            
        # Create timestamps based on sampling rate
        # Use 5-minute intervals as in preprocessing
        freq = '5min'
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        # Make timezone-aware
        start_time = start_time.replace(tzinfo=None)  # Make timezone-naive
        timestamps = pd.date_range(start=start_time, periods=len(values), freq=freq)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values,
            'mask': mask
        })
        
        # Keep all rows; downstream uses mask to handle missingness
        df = df.reset_index(drop=True)
        
        return df if len(df) > 0 else None
    
    # Removed: resample_and_sync_streams — preprocessing already aligns streams via interpolate_downsample_pad
