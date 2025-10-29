"""
Example Usage Script
===================

This script demonstrates how to use the AI-READI stress prediction pipeline
with different configurations and model types.
"""

from stress_pipeline import PipelineConfig, PreprocessingConfig
from main_pipeline import StressPredictionPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_cnn_example():
    """Example using CNN model"""
    logger.info("Running CNN example...")
    
    config = PipelineConfig(
        data_root="/Users/puskarkafle/Documents/Research/AI-READI",
        output_dir="/Users/puskarkafle/Documents/Research/stress_pipeline_output_cnn",
        model_type="cnn",
        window_length_min=10,
        stride_min=2,
        batch_size=16,
        num_epochs=50
    )
    
    pipeline = StressPredictionPipeline(config)
    pipeline.run_full_pipeline(participant_limit=10)

def run_transformer_example():
    """Example using Transformer model"""
    logger.info("Running Transformer example...")
    
    config = PipelineConfig(
        data_root="/Users/puskarkafle/Documents/Research/AI-READI",
        output_dir="/Users/puskarkafle/Documents/Research/stress_pipeline_output_transformer",
        model_type="transformer",
        window_length_min=15,
        stride_min=3,
        batch_size=8,
        num_epochs=30,
        hidden_dim=256,
        num_layers=4
    )
    
    pipeline = StressPredictionPipeline(config)
    pipeline.run_full_pipeline(participant_limit=5)

def run_multimodal_example():
    """Example using Multimodal model"""
    logger.info("Running Multimodal example...")
    
    config = PipelineConfig(
        data_root="/Users/puskarkafle/Documents/Research/AI-READI",
        output_dir="/Users/puskarkafle/Documents/Research/stress_pipeline_output_multimodal",
        model_type="multimodal",
        window_length_min=20,
        stride_min=5,
        batch_size=4,
        num_epochs=20
    )
    
    pipeline = StressPredictionPipeline(config)
    pipeline.run_full_pipeline(participant_limit=3)

def run_custom_configuration():
    """Example with custom configuration"""
    logger.info("Running custom configuration example...")
    
    config = PipelineConfig(
        data_root="/Users/puskarkafle/Documents/Research/AI-READI",
        output_dir="/Users/puskarkafle/Documents/Research/stress_pipeline_output_custom",
        model_type="tcn",
        window_length_min=5,
        stride_min=1,
        sampling_rate=2.0,  # 2 Hz sampling
        batch_size=32,
        learning_rate=5e-4,
        num_epochs=100,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        stress_threshold=0.6,
        alignment_tolerance_min=3
    )
    
    pipeline = StressPredictionPipeline(config)
    pipeline.run_full_pipeline(participant_limit=15)

def run_preprocessing_example():
    """Example using preprocessing-based data ingestion"""
    logger.info("Running preprocessing-based example...")
    
    config = PreprocessingConfig(
        data_root="/Users/puskarkafle/Documents/Research/AI-READI",
        output_dir="/Users/puskarkafle/Documents/Research/stress_pipeline_output_preprocessing",
        model_type="cnn",
        window_length_min=10,
        stride_min=2,
        batch_size=16,
        num_epochs=30,
        use_preprocessing=True
    )
    
    pipeline = StressPredictionPipeline(config)
    pipeline.run_full_pipeline(participant_limit=3)

def run_enhanced_preprocessing_example():
    """Example using enhanced preprocessing with caching and batch processing"""
    logger.info("Running enhanced preprocessing example...")
    
    config = PreprocessingConfig(
        data_root="/Users/puskarkafle/Documents/Research/AI-READI",
        output_dir="/Users/puskarkafle/Documents/Research/enhanced_preprocessing_output",
        model_type="cnn",
        window_length_min=10,
        stride_min=2,
        batch_size=16,
        num_epochs=20,
        use_preprocessing=True
    )
    
    pipeline = StressPredictionPipeline(config)
    pipeline.run_full_pipeline(participant_limit=5)

def run_batch_preprocessing_example():
    """Example of batch preprocessing all participants"""
    logger.info("Running batch preprocessing example...")
    
    from enhanced_preprocessing_data_ingestion import EnhancedPreprocessingDataIngestion
    
    config = PreprocessingConfig(
        data_root="/Users/puskarkafle/Documents/Research/AI-READI",
        output_dir="/Users/puskarkafle/Documents/Research/batch_preprocessing_output",
        use_preprocessing=True
    )
    
    preprocessing = EnhancedPreprocessingDataIngestion(config)
    
    # Get available participants
    participants = preprocessing.get_available_participants()
    logger.info(f"Found {len(participants)} participants")
    
    # Process first 10 participants as example
    example_participants = participants[:10]
    results = preprocessing.batch_preprocess_participants(example_participants, batch_size=5)
    
    # Print statistics
    stats = preprocessing.get_preprocessing_stats()
    logger.info(f"Preprocessing stats: {stats}")

if __name__ == "__main__":
    # Run different examples
    try:
        # Start with enhanced preprocessing example
        run_enhanced_preprocessing_example()
        
        # Uncomment to run other examples
        # run_preprocessing_example()
        # run_batch_preprocessing_example()
        # run_cnn_example()
        # run_transformer_example()
        # run_multimodal_example()
        # run_custom_configuration()
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        raise
