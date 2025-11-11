# 1D CNN Stress Prediction - Project Overview

## Project Structure

### Essential Files

#### Core Modules
- `stress_pipeline.py` - Pipeline configuration (PipelineConfig)
- `enhanced_preprocessing_data_ingestion.py` - Data loading with caching
- `windowing.py` - Windowing system for time-series data
- `data_splits.py` - Data splitting and DataLoader creation
- `models.py` - Model definitions (StressCNN, etc.)
- `evaluation.py` - Evaluation metrics and analysis

#### Main Scripts
- `train_improved_v2_1d_cnn.py` - Main training script
- `evaluate_improved_model.py` - Evaluation script

#### Preprocessing
- `preprocess/cgm_loader.py` - CGM data loading
- `preprocess/wearable_loader.py` - Wearable data loading

### Data Directories
- `AI-READI/` - Source data directory (903 participants)
- `final_training/preprocessed_cache/` - Cached preprocessed data (903 files)
- `final_training/results_improved_v2/` - Training results and models

### Configuration
- `requirements.txt` - Python dependencies

## Quick Start

### 1. Training

```bash
python train_improved_v2_1d_cnn.py
```

This will:
- Load data from cache (903 participants)
- Create windows (1-hour windows, 60-minute stride)
- Normalize labels
- Train 1D CNN model
- Save best model to `final_training/results_improved_v2/models/best_model.pt`

### 2. Evaluation

```bash
python evaluate_improved_model.py
```

This will:
- Load trained model
- Evaluate on train/val/test sets
- Generate metrics and predictions
- Save results to `final_training/results_improved_v2/`

## Model Configuration

### Architecture
- **Model Type**: 1D CNN
- **Hidden Dimension**: 256
- **Number of Layers**: 6
- **Dropout**: 0.3
- **Output Layer**: Two-layer (256→128→1)

### Training Configuration
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Epochs**: 150 (with early stopping)
- **Early Stopping Patience**: 25
- **Weight Decay**: 5e-4
- **Loss Function**: MSE Loss
- **Label Normalization**: Enabled

### Data Configuration
- **Window Length**: 60 minutes
- **Stride**: 60 minutes
- **Sampling Rate**: 5-minute intervals
- **Cache Location**: `final_training/preprocessed_cache/`

## Current Results

### Test Set Performance
- **RMSE**: 25.75
- **MAE**: 18.07
- **R²**: -0.09
- **Pearson Correlation**: 0.67
- **Spearman Correlation**: 0.65

### Improvements Over Baseline
- **RMSE**: -15.62% improvement
- **MAE**: -19.72% improvement
- **R²**: +0.44 improvement
- **Correlation**: +6.35% improvement

## Cache Configuration

The project uses a fixed cache location:
```
final_training/preprocessed_cache/
```

- All training scripts use this cache
- Missing files are automatically preprocessed and cached
- Cache contains 903 preprocessed participant files

See `CACHE_CONFIGURATION.md` for more details.

## File Structure

```
Research/
├── train_improved_v2_1d_cnn.py      # Main training script
├── evaluate_improved_model.py        # Evaluation script
├── stress_pipeline.py                # Configuration
├── enhanced_preprocessing_data_ingestion.py  # Data loading
├── windowing.py                      # Windowing
├── data_splits.py                    # Data splitting
├── models.py                         # Model definitions
├── evaluation.py                     # Evaluation metrics
├── preprocess/                       # Preprocessing modules
│   ├── cgm_loader.py
│   └── wearable_loader.py
├── AI-READI/                         # Source data
├── final_training/                   # Training outputs
│   ├── preprocessed_cache/           # Cached data (903 files)
│   ├── results_improved_v2/          # Current results
│   │   ├── models/best_model.pt      # Trained model
│   │   ├── label_scaler.pkl          # Label scaler
│   │   ├── predictions.csv           # Predictions
│   │   ├── window_level_metrics.json # Metrics
│   │   └── training_metrics.json     # Training metrics
│   └── logs/                         # Training logs
└── requirements.txt                  # Dependencies
```

## Monitoring Training

```bash
# Watch training progress
tail -f final_training/logs/improved_v2_training.log | grep -E "Epoch|Train Loss|Val Loss|Saved best"

# Check status
tail -20 final_training/logs/improved_v2_training.log
```

## Evaluation Results

After training, evaluation results are saved to:
- `final_training/results_improved_v2/window_level_metrics.json` - All metrics
- `final_training/results_improved_v2/predictions.csv` - Predictions

## Notes

- The model uses label normalization (StandardScaler on stress labels)
- All features are normalized using StandardScaler
- Participant-based train/val/test splits are used
- Cache is automatically managed (loads from cache if available)

## Troubleshooting

### Cache Issues
- Cache location: `final_training/preprocessed_cache/`
- If cache is missing, files will be automatically preprocessed
- Cache contains 903 participant files

### Model Loading
- Model path: `final_training/results_improved_v2/models/best_model.pt`
- Label scaler: `final_training/results_improved_v2/label_scaler.pkl`
- Ensure model configuration matches training configuration

