"""
Example Usage Script (1D CNN Only)
=================================

This script demonstrates how to run the AI-READI stress prediction pipeline
using only the 1D CNN model.
"""

from stress_pipeline import PipelineConfig
from main_pipeline import StressPredictionPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_cnn_example():
    """Run pipeline with 1D CNN model"""
    logger.info("Running CNN example...")
    
    config = PipelineConfig(
        data_root="/Users/puskarkafle/Documents/Research/AI-READI",
        output_dir="/Users/puskarkafle/Documents/Research/stress_pipeline_output_cnn",
        model_type="cnn",
        window_length_min=10,
        stride_min=2,
        batch_size=16,
        num_epochs=100
    )
    
    pipeline = StressPredictionPipeline(config)
    pipeline.run_full_pipeline(participant_limit=100)

if __name__ == "__main__":
    try:
        run_cnn_example()
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        raise
