# 1D CNN Stress Prediction Model

A deep learning model for predicting stress levels from multimodal physiological signals using a 1D Convolutional Neural Network.

## Project Structure

```
Research/
├── src/                          # Core modules
│   ├── __init__.py
│   ├── config.py                # Pipeline configuration
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── windowing.py              # Time-series windowing
│   ├── data_splits.py            # Data splitting and DataLoaders
│   ├── models.py                 # Model definitions (1D CNN)
│   └── evaluation.py             # Evaluation metrics
├── scripts/                      # Main scripts
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
├── preprocess/                   # Preprocessing modules
│   ├── __init__.py
│   ├── cgm_loader.py            # CGM data loading
│   └── wearable_loader.py      # Wearable data loading
├── data/                         # Data directories
│   ├── raw/                     # Raw data (AI-READI)
│   └── processed/                # Processed data and results
│       ├── preprocessed_cache/  # Cached preprocessed data (903 files)
│       ├── results_improved_v2/ # Training results
│       │   ├── models/          # Trained models
│       │   ├── predictions.csv # Predictions
│       │   └── *.json          # Metrics
│       └── logs/                # Training logs
├── docs/                         # Documentation
│   ├── readme_1d_cnn.md        # Detailed 1D CNN guide
│   ├── cache_configuration.md  # Cache configuration
│   ├── improvements_v2.md      # Model improvements
│   └── project_cleanup_summary.md
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python scripts/train.py
```

This will:
- Load data from cache (`data/processed/preprocessed_cache/`)
- Create 1-hour windows with 60-minute stride
- Normalize labels and features
- Train 1D CNN model (256 hidden dim, 6 layers)
- Save best model to `data/processed/results_improved_v2/models/best_model.pt`

### Evaluation

```bash
python scripts/evaluate.py
```

This will:
- Load the trained model
- Evaluate on train/val/test sets
- Generate comprehensive metrics
- Save results to `data/processed/results_improved_v2/`

## Model Architecture

- **Type**: 1D Convolutional Neural Network
- **Hidden Dimension**: 256
- **Number of Layers**: 6
- **Dropout**: 0.3
- **Output Layer**: Two-layer (256→128→1)

## Training Configuration

- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Epochs**: 150 (with early stopping)
- **Early Stopping Patience**: 25
- **Weight Decay**: 5e-4
- **Loss Function**: MSE Loss
- **Label Normalization**: Enabled

## Data Configuration

- **Window Length**: 60 minutes
- **Stride**: 60 minutes
- **Sampling Rate**: 5-minute intervals
- **Cache Location**: `data/processed/preprocessed_cache/`
- **Participants**: 903

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

## Cache Management

The project uses a fixed cache location:
```
data/processed/preprocessed_cache/
```

- All scripts automatically use this cache
- Missing files are automatically preprocessed and cached
- Cache contains 903 preprocessed participant files

See `docs/cache_configuration.md` for more details.

## Monitoring Training

```bash
# Watch training progress
tail -f data/processed/logs/improved_v2_training.log | grep -E "Epoch|Train Loss|Val Loss|Saved best"

# Check status
tail -20 data/processed/logs/improved_v2_training.log
```

## File Naming

### Core Modules (src/)
- `config.py` - Configuration (was `stress_pipeline.py`)
- `data_loader.py` - Data loading (was `enhanced_preprocessing_data_ingestion.py`)
- `windowing.py` - Windowing system
- `data_splits.py` - Data splitting
- `models.py` - Model definitions
- `evaluation.py` - Evaluation metrics

### Scripts (scripts/)
- `train.py` - Training script (was `train_improved_v2_1d_cnn.py`)
- `evaluate.py` - Evaluation script (was `evaluate_improved_model.py`)

## Dependencies

See `requirements.txt` for full list. Main dependencies:
- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0
- Matplotlib >= 3.4.0

## Documentation

- `docs/readme_1d_cnn.md` - Detailed 1D CNN guide
- `docs/cache_configuration.md` - Cache configuration
- `docs/improvements_v2.md` - Model improvements

## License

See LICENSE file for details.
