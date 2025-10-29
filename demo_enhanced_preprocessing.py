"""
Enhanced Preprocessing Integration Demo
=====================================

This script demonstrates the enhanced preprocessing integration with the preprocess folder.
It shows caching, batch processing, and full pipeline integration.
"""

import logging
from pathlib import Path
import time

from stress_pipeline import PreprocessingConfig
from enhanced_preprocessing_data_ingestion import EnhancedPreprocessingDataIngestion
from main_pipeline import StressPredictionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_enhanced_preprocessing():
    """Demonstrate enhanced preprocessing capabilities"""
    
    print("="*60)
    print("ENHANCED PREPROCESSING INTEGRATION DEMO")
    print("="*60)
    
    # Configuration
    config = PreprocessingConfig(
        data_root="/Users/puskarkafle/Documents/Research/AI-READI",
        output_dir="/Users/puskarkafle/Documents/Research/demo_enhanced_preprocessing",
        model_type="cnn",
        window_length_min=10,
        stride_min=2,
        batch_size=16,
        num_epochs=3,  # Short demo run
        use_preprocessing=True
    )
    
    print(f"Configuration:")
    print(f"  Data root: {config.data_root}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Model type: {config.model_type}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Use preprocessing: {config.use_preprocessing}")
    print()
    
    # Step 1: Initialize enhanced preprocessing
    print("Step 1: Initializing Enhanced Preprocessing...")
    preprocessing = EnhancedPreprocessingDataIngestion(config)
    
    # Step 2: Get available participants
    print("Step 2: Getting Available Participants...")
    participants = preprocessing.get_available_participants()
    print(f"Found {len(participants)} participants")
    print(f"First 10 participants: {participants[:10]}")
    print()
    
    # Step 3: Test caching with single participant
    print("Step 3: Testing Caching...")
    test_participant = participants[0]
    
    # First load (should process and cache)
    print(f"First load of participant {test_participant}...")
    start_time = time.time()
    streams1 = preprocessing.load_participant_streams(test_participant, use_cache=True)
    first_load_time = time.time() - start_time
    print(f"  Loaded streams: {list(streams1.keys())}")
    print(f"  Time taken: {first_load_time:.2f} seconds")
    
    # Second load (should use cache)
    print(f"Second load of participant {test_participant}...")
    start_time = time.time()
    streams2 = preprocessing.load_participant_streams(test_participant, use_cache=True)
    second_load_time = time.time() - start_time
    print(f"  Loaded streams: {list(streams2.keys())}")
    print(f"  Time taken: {second_load_time:.2f} seconds")
    print(f"  Speedup: {first_load_time/second_load_time:.1f}x")
    print()
    
    # Step 4: Test batch processing
    print("Step 4: Testing Batch Processing...")
    batch_participants = participants[:5]
    print(f"Processing {len(batch_participants)} participants in batches...")
    
    start_time = time.time()
    batch_results = preprocessing.batch_preprocess_participants(batch_participants, batch_size=3)
    batch_time = time.time() - start_time
    
    print(f"Batch processing completed in {batch_time:.2f} seconds")
    print(f"Successfully processed {len(batch_results)} participants")
    print()
    
    # Step 5: Show preprocessing statistics
    print("Step 5: Preprocessing Statistics...")
    stats = preprocessing.get_preprocessing_stats()
    print(f"Total participants found: {stats['total_participants']}")
    print(f"Successful loads: {stats['successful_loads']}")
    print(f"Failed loads: {stats['failed_loads']}")
    print(f"Cached loads: {stats['cached_loads']}")
    print()
    
    # Step 6: Run full pipeline with enhanced preprocessing
    print("Step 6: Running Full Pipeline with Enhanced Preprocessing...")
    print("This will demonstrate the complete integration...")
    
    pipeline = StressPredictionPipeline(config)
    pipeline.run_full_pipeline(participant_limit=3)
    
    print("Full pipeline completed successfully!")
    print()
    
    # Step 7: Show output files
    print("Step 7: Generated Files...")
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        files = list(output_dir.rglob("*"))
        print(f"Generated {len(files)} files in {output_dir}")
        
        # Show key files
        key_files = [
            "models/best_model.pt",
            "preprocessed_cache",
            "training_history.json",
            "preprocessing_summary.json"
        ]
        
        for key_file in key_files:
            file_path = output_dir / key_file
            if file_path.exists():
                if file_path.is_dir():
                    count = len(list(file_path.rglob("*")))
                    print(f"  {key_file}/ ({count} files)")
                else:
                    size = file_path.stat().st_size
                    print(f"  {key_file} ({size} bytes)")
    
    print()
    print("="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print()
    print("Key Benefits Demonstrated:")
    print("✓ Seamless integration with preprocess folder")
    print("✓ Automatic caching for faster subsequent runs")
    print("✓ Batch processing for efficiency")
    print("✓ Comprehensive statistics tracking")
    print("✓ Robust error handling")
    print("✓ Complete pipeline integration")
    print()
    print("Next Steps:")
    print("1. Use batch_preprocessing.py for large-scale preprocessing")
    print("2. Enable caching for faster development iterations")
    print("3. Monitor preprocessing statistics for data quality")
    print("4. Scale up to more participants and longer training")

if __name__ == "__main__":
    try:
        demo_enhanced_preprocessing()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
