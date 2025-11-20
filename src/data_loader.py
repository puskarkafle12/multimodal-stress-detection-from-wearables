"""
Enhanced Preprocessing Integration
================================

This module provides enhanced integration with the preprocess folder,
including batch processing, caching, and improved data handling.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pickle
import hashlib

# Add preprocess folder to path
preprocess_path = Path(__file__).parent.parent / "preprocess"
sys.path.insert(0, str(preprocess_path))

# Import preprocessing methods
from wearable_loader import (
    load_heartrate, load_stress, load_oxygen, load_resp, 
    load_activity, load_calories, load_sleep
)
from cgm_loader import load_cgm

logger = logging.getLogger(__name__)

class EnhancedPreprocessingDataIngestion:
    """Enhanced data ingestion using preprocessing methods with caching and batch processing"""
    
    def __init__(self, config):
        self.config = config
        self.data_root = config.data_root
        self.sampling_rate = config.sampling_rate
        
        # Define modality paths
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
        
        # Define target lengths for each modality - Updated for TA requirements (2880 = 10 days at 5-min intervals)
        self.target_lengths = {
            'heart_rate': 2880,  # 2880 values = 10 days at 5-min intervals
            'stress': 2880,
            'oxygen_saturation': 2880,
            'respiratory_rate': 2880,
            'physical_activity': 2880,
            'physical_activity_calorie': 2880,
            'sleep': 2880,  # Upsampled to 2880
            'cgm': 2880  # 2880 values = 10 days at 5-min intervals
        }
        
        # Cache directory for preprocessed data - ALWAYS use the same cache location
        # This ensures all training runs use the same cached data
        if hasattr(config, 'cache_dir') and config.cache_dir:
            # Use explicitly specified cache directory
            self.cache_dir = Path(config.cache_dir)
        else:
            # Default to the standard cache location (relative to project root)
            base_dir = Path(__file__).parent.parent
            self.cache_dir = base_dir / "data" / "processed" / "preprocessed_cache"
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache directory: {self.cache_dir}")
        
        # Cache version - changes when preprocessing code changes
        # This ensures old cache is invalidated when preprocessing logic changes
        self.cache_version = self._compute_preprocessing_version()
        logger.info(f"Preprocessing cache version: {self.cache_version}")
        
        # Preprocessing statistics
        self.stats = {
            'total_participants': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'cached_loads': 0
        }
        
    def _compute_preprocessing_version(self) -> str:
        """Compute version hash based on preprocessing code to invalidate cache when code changes"""
        try:
            # Hash the preprocessing files to detect code changes
            preprocess_dir = Path(__file__).parent.parent / "preprocess"
            files_to_hash = [
                preprocess_dir / "wearable_loader.py",
                preprocess_dir / "cgm_loader.py"
            ]
            
            combined_content = ""
            for file_path in files_to_hash:
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        combined_content += f.read()
            
            # Create hash from file contents
            version_hash = hashlib.md5(combined_content.encode()).hexdigest()[:8]
            return f"v{version_hash}"
        except Exception as e:
            logger.warning(f"Failed to compute preprocessing version: {e}")
            return "v_unknown"
    
    def get_available_participants(self) -> List[str]:
        """Get list of available participants with actual data files"""
        available_participants = []
        
        # Check heart rate data as a proxy for available participants
        heart_rate_path = self.modality_paths['heart_rate']
        if os.path.exists(heart_rate_path):
            participant_dirs = [d for d in os.listdir(heart_rate_path) if os.path.isdir(os.path.join(heart_rate_path, d))]
            available_participants = sorted(participant_dirs)
        
        self.stats['total_participants'] = len(available_participants)
        return available_participants
    
    def load_participant_streams(self, participant_id: str, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load all available data streams for a participant using preprocessing methods
        
        Args:
            participant_id: Participant identifier (e.g., "1023")
            use_cache: Whether to use cached preprocessed data
            
        Returns:
            Dictionary mapping modality names to DataFrames with columns [timestamp, value, mask]
        """
        # Check cache first
        cache_file = self.cache_dir / f"{participant_id}_preprocessed.pkl"
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check cache version - invalidate if version doesn't match
                cached_version = cached_data.get('_cache_version', 'v_old')
                if cached_version != self.cache_version:
                    logger.info(f"Cache version mismatch for {participant_id} (cached: {cached_version}, current: {self.cache_version}). Regenerating...")
                    # Delete old cache file
                    cache_file.unlink()
                else:
                    # Remove version metadata before returning
                    cached_data.pop('_cache_version', None)
                    logger.info(f"Loaded cached data for participant {participant_id}")
                    self.stats['cached_loads'] += 1
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache for {participant_id}: {e}")
        
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
        
        for modality, file_path in file_paths.items():
            try:
                if os.path.exists(file_path):
                    values, mask = self._load_with_preprocessing(modality, file_path)
                    
                    if values is not None and mask is not None:
                        stream_data = self._create_stream_dataframe(values, mask, modality)
                        if stream_data is not None:
                            streams[modality] = stream_data
                            logger.info(f"Loaded {modality} for participant {participant_id}: {len(stream_data)} samples")
                            
            except Exception as e:
                logger.warning(f"Failed to load {modality} for participant {participant_id}: {e}")
                self.stats['failed_loads'] += 1
                
        # Cache the results with version metadata
        if use_cache and streams:
            try:
                # Add cache version to streams dict before caching
                streams_with_version = streams.copy()
                streams_with_version['_cache_version'] = self.cache_version
                with open(cache_file, 'wb') as f:
                    pickle.dump(streams_with_version, f)
                logger.info(f"Cached preprocessed data for participant {participant_id} (version: {self.cache_version})")
            except Exception as e:
                logger.warning(f"Failed to cache data for {participant_id}: {e}")
        
        if streams:
            self.stats['successful_loads'] += 1
        else:
            self.stats['failed_loads'] += 1
            
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
        # Make timezone-naive
        start_time = start_time.replace(tzinfo=None)
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
    
    def batch_preprocess_participants(self, participant_ids: List[str], batch_size: int = 10) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Preprocess multiple participants in batches for efficiency
        
        Args:
            participant_ids: List of participant IDs to process
            batch_size: Number of participants to process in each batch
            
        Returns:
            Dictionary mapping participant_id to their preprocessed streams
        """
        all_results = {}
        
        for i in range(0, len(participant_ids), batch_size):
            batch = participant_ids[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: participants {batch[0]} to {batch[-1]}")
            
            for participant_id in batch:
                try:
                    streams = self.load_participant_streams(participant_id)
                    if streams:
                        all_results[participant_id] = streams
                except Exception as e:
                    logger.error(f"Failed to process participant {participant_id}: {e}")
        
        return all_results
    
    def get_preprocessing_stats(self) -> Dict[str, int]:
        """Get statistics about preprocessing performance"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear all cached preprocessed data"""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cleared preprocessing cache")
    
    def save_preprocessing_summary(self, output_path: str):
        """Save preprocessing summary to file"""
        summary = {
            'preprocessing_stats': self.stats,
            'target_lengths': self.target_lengths,
            'modality_paths': self.modality_paths,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved preprocessing summary to {output_path}")
