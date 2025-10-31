### AI-READI Wearables: Data Preparation and 1D CNN Training Guide

This guide explains how data is prepared, how the 1D CNN model is trained, and how results are validated in this repository. It reflects the current codebase behavior and focuses only on the 1D CNN path.

---

### 1) Data sources and ingestion

- **Dataset root**: `AI-READI/` in the workspace. The pipeline expects modality JSON files under `wearable_activity_monitor/*/garmin_vivosmart5/<participant_id>/` and CGM JSON files under `wearable_blood_glucose/.../dexcom_g6/<participant_id>/`.
- **Participants metadata**: `AI-READI/participants.tsv` with a `participant_id` and a `recommended_split` column used to assign train/val/test splits.
- **Ingestion implementation**: `EnhancedPreprocessingDataIngestion` in `enhanced_preprocessing_data_ingestion.py` loads available modalities per participant and returns a dict of DataFrames per modality.
  - Each DataFrame has columns: `timestamp`, `value`, `mask`.
  - `mask` is 1 if a value was observed (after preprocessing alignment), otherwise 0.

---

### 2) Preprocessing and alignment (uniform sequences)

- The repository reuses preprocessing utilities under `preprocess/` to create uniform, fixed-length time series per modality using 5-minute granularity.
- Key function: `interpolate_downsample_pad` in `preprocess/wearable_loader.py` and `preprocess/cgm_loader.py`:
  - Builds a regular timeline at a fixed 5-minute frequency.
  - Places known observations onto that timeline.
  - Interpolates missing values linearly and performs forward/backward fill when needed.
  - Downsamples if the raw series is longer than the target length.
  - Pads with zeros if shorter.
  - Returns both the processed `values` and a `mask` indicating where data originally existed.
- Target lengths per modality are defined in `EnhancedPreprocessingDataIngestion.target_lengths` to ensure consistent shapes.

Result: Each modality becomes a time-aligned, fixed-length sequence with an accompanying mask. We keep all rows (including masked) so downstream logic can handle missingness consistently.

---

### 3) Harmonization and windowing

- After ingestion, streams are already aligned by preprocessing. No further resampling is performed.
- Windowing is handled by `WindowingSystem` in `windowing.py`:
  - Computes window length and stride in samples using `window_length_min`, `stride_min`, and `sampling_rate` from `PipelineConfig`.
  - Slides over the synchronized timestamps to create many fixed-size windows.
  - Builds a window feature tensor of shape `[T, F]` where `T = window_length_samples` and `F = 2 * (#modalities)` because each modality contributes two channels: `value` and `mask`.
  - If a modality is shorter than the current window’s end, it pads the tail of the window with zeros (both values and mask).
  - Aligns the window’s label by finding the closest stress value to the window center time within `alignment_tolerance_min` minutes, requiring mask = 1.

Why windows?
- Models require fixed-size tensors; windows provide uniform [T, F] blocks.
- They preserve short-term temporal patterns and greatly increase the number of training examples per participant.

---

### 4) Train/val/test splits and scaling

- Splits are participant-based using `DataSplitter` in `data_splits.py`:
  - Reads `recommended_split` from `participants.tsv` to assign participant windows to train/val/test.
  - Creates PyTorch DataLoaders for each split.
- Feature scaling:
  - A `StandardScaler` is fit on the flattened feature vectors from the training windows only.
  - The same scaler is applied to val and test windows to avoid data leakage.

Tips to ensure non-empty splits:
- Use a `participants.tsv` that includes at least one participant in each of train/val/test.
- When testing interactively with `--participant_limit`, you may get empty val/test sets; that is expected. The pipeline will skip downstream steps that need those sets.

---

### 5) 1D CNN model

- Implementation: `StressCNN` in `models.py`.
- Expected input shape at forward pass: `[batch, window_length, input_dim]`.
- Architecture outline:
  - Stack of 1D Conv layers with BatchNorm, ReLU, and Dropout.
  - Global average pooling over time.
  - Final linear layer to produce a single regression output per window.
- Model is created via `ModelFactory.create_model(model_type="cnn", ...)` using hyperparameters from `PipelineConfig`:
  - `hidden_dim`, `num_layers`, `dropout`, and the derived `window_length`.

---

### 6) Training loop, saving, and evaluation

- Training is managed in `training.py` via `StressTrainer` (for non-multimodal types):
  - Uses MSE (or equivalent) to regress stress value per window.
  - Early stopping based on validation loss.
  - Saves the best model to `output_dir/models/best_model.pt`.
- Inference and evaluation (window-level, participant-level, subgroup, etc.) are handled by `inference.py` and `evaluation.py` when there is a non-empty test set.
- Interpretability and robustness steps (`interpretability.py`) are also skipped if no test data is available.

---

### 7) Running the pipeline (1D CNN only)

Minimal example (single participant sanity check):
```bash
python3 main_pipeline.py \
  --data_root /Users/puskarkafle/Documents/Research/AI-READI \
  --output_dir /Users/puskarkafle/Documents/Research/stress_pipeline_output \
  --model_type cnn \
  --window_length 10 \
  --stride 2 \
  --batch_size 32 \
  --epochs 50 \
  --participant_limit 1
```

Typical full run (ensure `participants.tsv` covers all three splits):
```bash
python3 main_pipeline.py \
  --data_root /Users/puskarkafle/Documents/Research/AI-READI \
  --output_dir /Users/puskarkafle/Documents/Research/stress_pipeline_output \
  --model_type cnn \
  --window_length 10 \
  --stride 2 \
  --batch_size 64 \
  --epochs 100
```

Outputs:
- Best model: `stress_pipeline_output/models/best_model.pt`
- Evaluation plots and metrics: `stress_pipeline_output/plots/` and `stress_pipeline_output/results/` (when test data exists)
- Logs: `stress_pipeline.log`

---

### 8) Reproducibility and configuration

- Core configuration lives in `stress_pipeline.PipelineConfig`. Key fields for CNN training:
  - `window_length_min`, `stride_min`, `sampling_rate`
  - `batch_size`, `num_epochs`, `learning_rate`
  - `hidden_dim`, `num_layers`, `dropout`
- The output directory is created if it doesn’t exist and subfolders for `models/`, `results/`, and `plots/` are prepared automatically.

---

### 9) Troubleshooting

- "No data could be processed" after ingestion:
  - Ensure files exist for the targeted participant and modality paths.
  - Check permissions and JSON validity.
- Empty val/test splits:
  - Confirm `participants.tsv` includes `recommended_split` entries across train/val/test.
  - Avoid very small `--participant_limit` if you want non-empty validation/test.
- Shape mismatches during window feature concatenation:
  - This code pads per-window slices for short modalities. If you fork or modify windowing, ensure all modalities contribute `[T, 2]` consistently.

---

### 10) Quick summary of the data flow

1. Load per-modality JSONs → preprocess with `interpolate_downsample_pad` → fixed-length sequences + masks.
2. Build per-modality DataFrames with `timestamp`, `value`, `mask` (keep all rows; mask marks missingness).
3. Create sliding windows → `[T, F]` feature tensors combining all modalities (value + mask channels).
4. Align stress label at window center time.
5. Split by participant (train/val/test) and scale features using only train fit.
6. Train 1D CNN; early stop; save best model; evaluate/plot when test data exists.


