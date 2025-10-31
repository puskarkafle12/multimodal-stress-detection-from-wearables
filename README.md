# AI-READI Wearable Stress Prediction Pipeline

A comprehensive pipeline for preprocessing wearable data and predicting stress levels from multimodal physiological signals in the AI-READI dataset.

## Overview

This pipeline implements a complete machine learning workflow for stress prediction from wearable sensor data, including:

1. **Data Ingestion & Harmonization**: Load and synchronize multimodal wearable streams
2. **Windowing & Label Alignment**: Create sliding windows and align stress labels
3. **Participant-based Splits**: Train/validation/test splits to prevent data leakage
4. **Multiple Model Architectures**: CNN, TCN, Transformer, and Multimodal models
5. **Training with Regularization**: Early stopping, group-aware training, temporal smoothness
6. **Inference & Aggregation**: Participant-level stress metrics and patterns
7. **Comprehensive Evaluation**: Window-level, participant-level, temporal, and subgroup analysis
8. **Interpretability & Robustness**: Feature importance, ablation studies, robustness testing

## Features

### Data Processing
- **Multimodal Integration**: Heart rate, SpO₂, respiratory rate, physical activity, sleep, CGM
- **Temporal Synchronization**: Resample all streams to common sampling rate
- **Artifact Handling**: Physiological range clipping and missing data masks
- **Sliding Windows**: Configurable window length and stride

### Model Architectures
- **1D CNN**: Convolutional neural network with global pooling
- **TCN**: Temporal Convolutional Network with dilated convolutions
- **Transformer**: Self-attention based encoder with positional encoding
- **Multimodal**: Separate encoders per modality with fusion layer

### Training Features
- **Early Stopping**: Prevent overfitting with validation monitoring
- **Group-aware Training**: Reduce subgroup bias across study groups
- **Temporal Regularization**: Encourage smooth predictions over time
- **Learning Rate Scheduling**: Adaptive learning rate reduction

### Evaluation Metrics
- **Window-level**: MAE, RMSE, R², F1, AUC, calibration error
- **Participant-level**: Mean stress, high-stress percentage, volatility
- **Temporal**: Serial correlation, autocorrelation analysis
- **Subgroup**: Performance across study groups, age, clinical sites

### Interpretability
- **Feature Importance**: Permutation, gradient, and integrated gradient methods
- **Modality Analysis**: Importance of each physiological modality
- **Ablation Studies**: Impact of removing features or modalities
- **Robustness Testing**: Performance under missing data, resolution changes

## Installation

1. Clone or download the pipeline files
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from stress_pipeline import PipelineConfig
from main_pipeline import StressPredictionPipeline

# Create configuration
config = PipelineConfig(
    data_root="/path/to/AI-READI",
    output_dir="/path/to/output",
    model_type="cnn",
    window_length_min=10,
    stride_min=2
)

# Run pipeline
pipeline = StressPredictionPipeline(config)
pipeline.run_full_pipeline()
```

### Command Line Usage

```bash
python main_pipeline.py \
    --data_root /path/to/AI-READI \
    --output_dir /path/to/output \
    --model_type cnn \
    --window_length 10 \
    --stride 2 \
    --batch_size 32 \
    --epochs 100
```

### Advanced Configuration (1D CNN)

```python
config = PipelineConfig(
    # Data paths
    data_root="/path/to/AI-READI",
    output_dir="/path/to/output",
    
    # Windowing parameters
    window_length_min=10,  # 10-minute windows
    stride_min=2,          # 2-minute stride
    sampling_rate=1.0,     # 1 Hz sampling
    
    # Model parameters (1D CNN)
    model_type="cnn",
    hidden_dim=128,
    num_layers=3,
    dropout=0.2,
    
    # Training parameters
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=100,
    early_stopping_patience=10,
    
    # Evaluation
    stress_threshold=0.5,
    alignment_tolerance_min=5
)
```

## Pipeline Components

### 1. Data Ingestion (`enhanced_preprocessing_data_ingestion.py`)
- Loads JSON data files for each participant
- Extracts time series data from different modalities
- Handles missing values and invalid readings
- Resamples and synchronizes all streams

### 2. Windowing System (`windowing.py`)
- Creates sliding windows with configurable length and stride
- Aligns stress labels to window centers
- Builds feature tensors with value and mask channels
- Computes proxy stress labels from physiological indicators

### 3. Data Splits (`data_splits.py`)
- Participant-based train/validation/test splits
- Prevents data leakage across participants
- PyTorch Dataset and DataLoader classes
- Feature normalization and scaling

### 4. Model Architectures (`models.py`)
- Multiple neural network architectures
- Custom loss functions (Huber, temporal smoothness, focal)
- Model factory for easy architecture selection

### 5. Training (`training.py`)
- Training loop with early stopping
- Group-aware regularization
- Learning rate scheduling
- Comprehensive monitoring and logging

### 6. Inference (`inference.py`)
- Participant-level prediction aggregation
- Diurnal pattern analysis
- Weekday/weekend comparisons
- Comprehensive visualization

### 7. Evaluation (`evaluation.py`)
- Window-level and participant-level metrics
- Temporal consistency analysis
- Subgroup performance evaluation
- Calibration analysis

### 8. Interpretability (`interpretability.py`)
- Feature and modality importance
- Robustness testing
- Ablation studies
- Comprehensive visualization


### Windowing Parameters
- `window_length_min`: Window length in minutes (default: 10)
- `stride_min`: Stride between windows in minutes (default: 2)
- `sampling_rate`: Target sampling rate in Hz (default: 1.0)

### Model Parameters
- `model_type`: Architecture type ("cnn", "tcn", "transformer", "multimodal")
- `hidden_dim`: Hidden dimension size (default: 128)
- `num_layers`: Number of layers (default: 3)
- `dropout`: Dropout rate (default: 0.2)

### Training Parameters
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 1e-3)
- `num_epochs`: Maximum epochs (default: 100)
- `early_stopping_patience`: Early stopping patience (default: 10)

### Evaluation Parameters
- `stress_threshold`: Threshold for high-stress classification (default: 0.5)
- `alignment_tolerance_min`: Label alignment tolerance in minutes (default: 5)
