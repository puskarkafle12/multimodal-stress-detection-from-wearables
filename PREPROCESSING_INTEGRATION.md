# AI-READI Stress Prediction Pipeline - Preprocessing Integration

This document explains how the pipeline integrates with the `preprocess` folder for enhanced data preprocessing.

## Overview

The pipeline now uses the preprocessing methods from the `preprocess` folder to:
- Load and process wearable sensor data
- Handle missing data with mask channels
- Resample data to consistent intervals
- Cache preprocessed data for faster subsequent runs
- Batch process multiple participants efficiently

## Key Components

### 1. Enhanced Preprocessing Data Ingestion (`enhanced_preprocessing_data_ingestion.py`)

This module provides:
- **Caching**: Preprocessed data is cached to avoid reprocessing
- **Batch Processing**: Process multiple participants efficiently
- **Statistics Tracking**: Monitor preprocessing performance
- **Error Handling**: Robust handling of missing or corrupted data

### 2. Batch Preprocessing Utility (`batch_preprocessing.py`)

Command-line utility for preprocessing all participants:
```bash
python batch_preprocessing.py \
    --data_root /path/to/AI-READI \
    --output_dir /path/to/output \
    --participant_limit 100 \
    --batch_size 10
```

### 3. Preprocessing Configuration (`stress_pipeline.py`)

The `PreprocessingConfig` class includes:
- Target lengths for each modality
- Sampling rate (0.2 Hz = 1 sample per 5 minutes)
- Preprocessing-specific parameters

## Data Flow

```
Raw JSON Files → Preprocess Functions → Enhanced Ingestion → Pipeline
     ↓                ↓                      ↓              ↓
  [1023_heartrate.json] → [load_heartrate()] → [Cached Data] → [Windows]
  [1023_stress.json]    → [load_stress()]    → [DataFrames] → [Training]
  [1023_sleep.json]     → [load_sleep()]     → [Statistics]  → [Evaluation]
```

## Usage Examples

### Basic Usage
```python
from stress_pipeline import PreprocessingConfig
from main_pipeline import StressPredictionPipeline

config = PreprocessingConfig(
    data_root="/path/to/AI-READI",
    output_dir="/path/to/output",
    use_preprocessing=True
)

pipeline = StressPredictionPipeline(config)
pipeline.run_full_pipeline(participant_limit=10)
```

### Batch Preprocessing
```python
from enhanced_preprocessing_data_ingestion import EnhancedPreprocessingDataIngestion

preprocessing = EnhancedPreprocessingDataIngestion(config)
participants = preprocessing.get_available_participants()
results = preprocessing.batch_preprocess_participants(participants[:50])
```

### Command Line Batch Processing
```bash
# Preprocess all participants
python batch_preprocessing.py \
    --data_root /Users/puskarkafle/Documents/Research/AI-READI \
    --output_dir /Users/puskarkafle/Documents/Research/preprocessed_data

# Preprocess first 100 participants
python batch_preprocessing.py \
    --data_root /Users/puskarkafle/Documents/Research/AI-READI \
    --output_dir /Users/puskarkafle/Documents/Research/preprocessed_data \
    --participant_limit 100

# Clear cache
python batch_preprocessing.py \
    --data_root /Users/puskarkafle/Documents/Research/AI-READI \
    --output_dir /Users/puskarkafle/Documents/Research/preprocessed_data \
    --clear_cache
```

## Preprocessing Methods Used

The pipeline integrates these functions from the `preprocess` folder:

### From `wearable_loader.py`:
- `load_heartrate()`: Process heart rate data
- `load_stress()`: Process stress data
- `load_oxygen()`: Process oxygen saturation data
- `load_resp()`: Process respiratory rate data
- `load_activity()`: Process physical activity data
- `load_calories()`: Process calorie data
- `load_sleep()`: Process sleep data

### From `cgm_loader.py`:
- `load_cgm()`: Process continuous glucose monitoring data

## Data Format

Each preprocessing function returns:
- `values`: NumPy array of processed values
- `mask`: NumPy array indicating data availability (1 = valid, 0 = missing)

The enhanced ingestion converts these to DataFrames with columns:
- `timestamp`: Time index
- `value`: Processed sensor value
- `mask`: Data availability mask

## Target Lengths

Each modality has a predefined target length:
- Heart Rate: 3000 samples
- Stress: 3000 samples
- Oxygen Saturation: 2411 samples
- Respiratory Rate: 3000 samples
- Physical Activity: 3000 samples
- Physical Activity Calorie: 1895 samples
- Sleep: 403 samples
- CGM: 2856 samples

## Caching

Preprocessed data is automatically cached in:
```
{output_dir}/preprocessed_cache/{participant_id}_preprocessed.pkl
```

Benefits:
- Faster subsequent runs
- Consistent preprocessing results
- Reduced computational load

## Statistics Tracking

The enhanced preprocessing tracks:
- Total participants found
- Successful loads
- Failed loads
- Cached loads
- Processing time

## Error Handling

The pipeline handles:
- Missing files gracefully
- Corrupted JSON data
- Empty data streams
- Cache corruption
- File permission issues

## Performance Tips

1. **Use Caching**: Enable caching for faster subsequent runs
2. **Batch Processing**: Process participants in batches for efficiency
3. **Monitor Statistics**: Check preprocessing stats for data quality
4. **Clear Cache**: Clear cache when preprocessing methods change
5. **Limit Participants**: Use `participant_limit` for testing

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `preprocess` folder is in Python path
2. **Missing Files**: Check that participant directories exist
3. **Cache Issues**: Clear cache if preprocessing methods change
4. **Memory Issues**: Reduce batch size for large datasets
5. **Permission Errors**: Ensure write access to output directory

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Main Pipeline

The enhanced preprocessing integrates seamlessly with:
- Windowing system
- Data splits
- Model training
- Evaluation
- Interpretability analysis

All components work together to provide a complete stress prediction pipeline using the preprocessing methods from the `preprocess` folder.
