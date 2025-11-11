# Cache Configuration - Fixed Location

## Overview
All training scripts now use a **fixed cache location** to ensure consistency and avoid duplicating preprocessed data.

## Fixed Cache Directory
```
/Users/puskarkafle/Documents/Research/final_training/results_improved/preprocessed_cache/
```

## How It Works

### 1. Cache Location
- **Always uses**: `/Users/puskarkafle/Documents/Research/final_training/results_improved/preprocessed_cache/`
- **Never creates**: New cache directories in output folders
- **Never copies**: Cache to other locations

### 2. Cache Loading Logic
1. **Check cache first**: For each participant, check if cached file exists
2. **Load from cache**: If found, load and return cached data
3. **Preprocess if missing**: If not found, preprocess the participant data
4. **Save to cache**: Save preprocessed data to the fixed cache location

### 3. Benefits
- ✅ **No duplication**: All scripts use the same cache
- ✅ **Faster loading**: Cached data loads instantly
- ✅ **Incremental processing**: Only missing participants are preprocessed
- ✅ **Consistency**: All training runs use the same preprocessed data

## Implementation

### Updated Files

1. **`enhanced_preprocessing_data_ingestion.py`**
   - Checks for `config.cache_dir` first
   - Defaults to fixed cache location if not set
   - Always uses the same cache directory

2. **`train_improved_v2_1d_cnn.py`**
   - Sets `config.cache_dir` to fixed location
   - Removed cache copying logic
   - Checks cache status before training

## Usage

### Setting Cache Directory in Training Scripts

```python
# FIXED cache directory - always use the same location
fixed_cache_dir = "/Users/puskarkafle/Documents/Research/final_training/results_improved/preprocessed_cache"

config = PipelineConfig(...)

# Set the fixed cache directory in config
config.cache_dir = fixed_cache_dir
```

### Cache Status

The training script will show:
```
✓ Using fixed cache directory: /Users/puskarkafle/Documents/Research/final_training/results_improved/preprocessed_cache
✓ Cache contains 903/903 files
  Missing files will be preprocessed and saved to cache
```

## Cache File Format

Each cached file:
- **Filename**: `{participant_id}_preprocessed.pkl`
- **Format**: Pickled dictionary with 8 modalities
- **Content**: Preprocessed DataFrames with columns `[timestamp, value, mask]`
- **Size**: ~465 KB per participant

## Current Cache Status

- **Total files**: 903/903 (100% complete)
- **Location**: `/Users/puskarkafle/Documents/Research/final_training/results_improved/preprocessed_cache/`
- **Last updated**: November 9, 2025 16:35:57

## Notes

- The cache directory is **shared** across all training runs
- New participants will be automatically preprocessed and cached
- Missing cache files will be created during training
- No need to manually manage cache locations

