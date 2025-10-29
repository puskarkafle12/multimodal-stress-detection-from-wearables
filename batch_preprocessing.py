"""
Batch Preprocessing Utility
=========================

This script provides utilities for batch preprocessing using the preprocess folder methods.
It can preprocess all participants and save the results for faster pipeline execution.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict
import json
import time

from stress_pipeline import PreprocessingConfig
from enhanced_preprocessing_data_ingestion import EnhancedPreprocessingDataIngestion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def batch_preprocess_all_participants(
    data_root: str,
    output_dir: str,
    participant_limit: int = None,
    batch_size: int = 10,
    use_cache: bool = True
) -> Dict[str, int]:
    """
    Preprocess all participants using the preprocess folder methods
    
    Args:
        data_root: Path to AI-READI dataset
        output_dir: Output directory for preprocessed data
        participant_limit: Maximum number of participants to process (None for all)
        batch_size: Number of participants to process in each batch
        use_cache: Whether to use cached data
        
    Returns:
        Dictionary with preprocessing statistics
    """
    
    # Create configuration
    config = PreprocessingConfig(
        data_root=data_root,
        output_dir=output_dir,
        use_preprocessing=True
    )
    
    # Initialize enhanced preprocessing
    preprocessing = EnhancedPreprocessingDataIngestion(config)
    
    # Get available participants
    all_participants = preprocessing.get_available_participants()
    
    if participant_limit:
        participants_to_process = all_participants[:participant_limit]
    else:
        participants_to_process = all_participants
    
    logger.info(f"Found {len(all_participants)} total participants")
    logger.info(f"Processing {len(participants_to_process)} participants")
    
    # Process participants in batches
    start_time = time.time()
    results = preprocessing.batch_preprocess_participants(
        participants_to_process, 
        batch_size=batch_size
    )
    end_time = time.time()
    
    # Get statistics
    stats = preprocessing.get_preprocessing_stats()
    stats['processing_time_seconds'] = end_time - start_time
    stats['participants_processed'] = len(results)
    
    # Save preprocessing summary
    summary_path = Path(output_dir) / "preprocessing_summary.json"
    preprocessing.save_preprocessing_summary(str(summary_path))
    
    logger.info(f"Preprocessing completed in {stats['processing_time_seconds']:.2f} seconds")
    logger.info(f"Successfully processed {stats['successful_loads']} participants")
    logger.info(f"Failed to process {stats['failed_loads']} participants")
    logger.info(f"Used cached data for {stats['cached_loads']} participants")
    
    return stats

def clear_preprocessing_cache(output_dir: str):
    """Clear all cached preprocessed data"""
    
    config = PreprocessingConfig(
        data_root="/dummy",  # Not used for cache clearing
        output_dir=output_dir,
        use_preprocessing=True
    )
    
    preprocessing = EnhancedPreprocessingDataIngestion(config)
    preprocessing.clear_cache()

def main():
    parser = argparse.ArgumentParser(description='Batch preprocessing utility for AI-READI dataset')
    
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to AI-READI dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for preprocessed data')
    parser.add_argument('--participant_limit', type=int, default=None,
                       help='Maximum number of participants to process (default: all)')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Number of participants to process in each batch (default: 10)')
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Use cached preprocessed data if available')
    parser.add_argument('--clear_cache', action='store_true',
                       help='Clear all cached preprocessed data')
    
    args = parser.parse_args()
    
    if args.clear_cache:
        logger.info("Clearing preprocessing cache...")
        clear_preprocessing_cache(args.output_dir)
        logger.info("Cache cleared successfully")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run batch preprocessing
    try:
        stats = batch_preprocess_all_participants(
            data_root=args.data_root,
            output_dir=args.output_dir,
            participant_limit=args.participant_limit,
            batch_size=args.batch_size,
            use_cache=args.use_cache
        )
        
        # Print final statistics
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Total participants found: {stats['total_participants']}")
        print(f"Participants processed: {stats['participants_processed']}")
        print(f"Successful loads: {stats['successful_loads']}")
        print(f"Failed loads: {stats['failed_loads']}")
        print(f"Cached loads: {stats['cached_loads']}")
        print(f"Processing time: {stats['processing_time_seconds']:.2f} seconds")
        print(f"Average time per participant: {stats['processing_time_seconds']/stats['participants_processed']:.2f} seconds")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Batch preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()
