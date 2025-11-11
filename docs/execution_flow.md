# Script Execution Flow

## Entry Points

### 1. Training Script: `scripts/train.py`

**Starting Point:**
```python
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
```

**Execution Flow:**

```
1. Script starts at line 494: `if __name__ == "__main__"`
   ↓
2. Calls `main()` function (line 165)
   ↓
3. main() function execution:
   
   Step 1: Setup Configuration (lines 174-210)
   - Sets up paths (output_dir, cache_dir)
   - Creates PipelineConfig
   - Checks cache status
   
   Step 2: Initialize Pipeline Components (lines 231-238)
   - EnhancedPreprocessingDataIngestion
   - WindowingSystem
   - DataSplitter
   - ModelFactory
   
   Step 3: Load Data (lines 240-270)
   - Loads all 903 participants from cache
   - Creates windows for each participant
   - Collects all windows
   
   Step 4: Create Data Splits (lines 272-280)
   - Participant-based train/val/test splits
   - Train: 32,805 windows
   - Val: 6,268 windows
   - Test: 6,728 windows
   
   Step 5: Normalize Labels (lines 282-290)
   - Creates label scaler from training data
   - Applies normalization to all splits
   
   Step 6: Create Data Loaders (lines 292-300)
   - Fits StandardScaler on training features
   - Creates PyTorch DataLoaders
   
   Step 7: Create Model (lines 302-320)
   - Creates 1D CNN model
   - Initializes weights
   - Sets output bias to 0
   
   Step 8: Train Model (lines 322-450)
   - Creates ImprovedStressTrainerV2
   - Runs training loop (150 epochs max)
   - Early stopping (patience: 25)
   - Saves best model
   
   Step 9: Final Evaluation (lines 452-480)
   - Evaluates on all splits
   - Saves metrics and predictions
```

### 2. Evaluation Script: `scripts/evaluate.py`

**Starting Point:**
```python
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
```

**Execution Flow:**

```
1. Script starts at line 321: `if __name__ == "__main__"`
   ↓
2. Calls `main()` function (line 291)
   ↓
3. main() function execution:
   
   Step 1: Setup Configuration (lines 295-310)
   - Creates PipelineConfig
   - Sets paths and cache directory
   
   Step 2: Load Model (lines 48-148)
   - Loads saved model checkpoint
   - Loads label scaler
   - Sets model to eval mode
   
   Step 3: Load Data (lines 82-120)
   - Loads all participants
   - Creates windows
   - Creates data splits
   - Applies feature scaling
   
   Step 4: Evaluate (lines 150-280)
   - Evaluates on train/val/test sets
   - Computes metrics (RMSE, MAE, R², correlation)
   - Saves predictions and metrics
```

## Key Functions

### Training Script (`scripts/train.py`)

1. **`main()`** (line 165) - Main entry point
2. **`create_label_scaler()`** (line 96) - Creates label normalizer
3. **`initialize_model_weights()`** (line 130) - Initializes model weights
4. **`ImprovedStressTrainerV2`** (line 43) - Training class

### Evaluation Script (`scripts/evaluate.py`)

1. **`main()`** (line 291) - Main entry point
2. **`evaluate_improved_model()`** (line 48) - Evaluation function

## How to Run

### Training:
```bash
python scripts/train.py
```

### Evaluation:
```bash
python scripts/evaluate.py
```

## Import Chain

### Training Script:
```
scripts/train.py
  ↓ imports
src/config.py (PipelineConfig)
src/data_loader.py (EnhancedPreprocessingDataIngestion)
src/windowing.py (WindowingSystem)
src/data_splits.py (DataSplitter)
src/models.py (ModelFactory)
```

### Evaluation Script:
```
scripts/evaluate.py
  ↓ imports
src/config.py (PipelineConfig)
src/data_loader.py (EnhancedPreprocessingDataIngestion)
src/windowing.py (WindowingSystem)
src/data_splits.py (DataSplitter)
src/models.py (ModelFactory)
src/evaluation.py (StressEvaluator)
```

## Execution Order Summary

### Training Pipeline:
1. **Configuration** → 2. **Data Loading** → 3. **Windowing** → 4. **Splitting** → 
5. **Normalization** → 6. **Model Creation** → 7. **Training** → 8. **Evaluation**

### Evaluation Pipeline:
1. **Configuration** → 2. **Model Loading** → 3. **Data Loading** → 
4. **Windowing** → 5. **Splitting** → 6. **Evaluation** → 7. **Save Results**

