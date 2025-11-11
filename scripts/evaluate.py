"""
Evaluate Improved Model with Label Normalization
=================================================
Evaluates the improved 1D CNN model with proper label scaling.
"""

import sys
from pathlib import Path
import torch
import json
import pandas as pd
import numpy as np
import logging
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add preprocess to path
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess"))

from config import PipelineConfig
from data_splits import DataSplitter
from evaluation import StressEvaluator
from data_loader import EnhancedPreprocessingDataIngestion
from windowing import WindowingSystem
from models import ModelFactory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def apply_label_scaling(windows, scaler):
    """Apply label scaling to windows"""
    if scaler is None:
        return windows
    
    import copy
    scaled_windows = []
    for window in windows:
        new_window = copy.deepcopy(window)
        if new_window.get('stress_label') is not None:
            original_label = new_window['stress_label']
            scaled_label = scaler.transform([[original_label]])[0][0]
            new_window['stress_label'] = float(scaled_label)
        scaled_windows.append(new_window)
    
    return scaled_windows

def evaluate_improved_model(config: PipelineConfig, results_dir: Path):
    """Evaluate improved model with label normalization"""
    
    model_path = results_dir / "models" / "best_model.pt"
    label_scaler_path = results_dir / "label_scaler.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    logger.info("=" * 80)
    logger.info("EVALUATING IMPROVED MODEL")
    logger.info("=" * 80)
    
    # Load label scaler
    label_scaler = None
    if label_scaler_path.exists():
        with open(label_scaler_path, 'rb') as f:
            label_scaler = pickle.load(f)
        logger.info("✓ Loaded label scaler")
    else:
        logger.warning("⚠ Label scaler not found, assuming no normalization")
    
    # Set cache directory in config
    base_dir = Path(__file__).parent.parent
    config.cache_dir = str(base_dir / "data" / "processed" / "preprocessed_cache")
    
    # Load data
    logger.info("Loading data...")
    data_ingestion = EnhancedPreprocessingDataIngestion(config)
    windowing_system = WindowingSystem(config)
    data_splitter = DataSplitter(config)
    
    # Load all windows
    all_windows = []
    participants = data_ingestion.get_available_participants()
    
    logger.info(f"Processing {len(participants)} participants...")
    for i, participant_id in enumerate(participants):
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(participants)} participants...")
        
        try:
            streams = data_ingestion.load_participant_streams(participant_id)
            if not streams:
                continue
            
            windows = windowing_system.create_windows(streams, participant_id)
            if windows:
                all_windows.extend(windows)
        except Exception as e:
            logger.warning(f"Error processing participant {participant_id}: {e}")
            continue
    
    logger.info(f"Total windows: {len(all_windows)}")
    
    # Create splits
    logger.info("Creating data splits...")
    participants_df = data_ingestion.participants_df
    train_windows, val_windows, test_windows = data_splitter.create_splits(
        all_windows, participants_df
    )
    
    logger.info(f"Train: {len(train_windows)}, Val: {len(val_windows)}, Test: {len(test_windows)}")
    
    # Apply label scaling if scaler exists
    if label_scaler is not None:
        logger.info("Applying label normalization...")
        train_windows = apply_label_scaling(train_windows, label_scaler)
        val_windows = apply_label_scaling(val_windows, label_scaler)
        test_windows = apply_label_scaling(test_windows, label_scaler)
        logger.info("✓ Labels normalized")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = data_splitter.create_dataloaders(
        train_windows, val_windows, test_windows
    )
    
    # Load model
    logger.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['features'].shape[2]
    
    model = ModelFactory.create_model(config.model_type, input_dim, config)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    logger.info("✓ Model loaded")
    
    # Initialize evaluator
    evaluator = StressEvaluator(str(results_dir))
    
    # Evaluate on all splits
    logger.info("Running evaluation...")
    metrics_output = {}
    
    def get_predictions_and_labels(model, loader, device, label_scaler):
        """Get predictions and labels, denormalizing if needed"""
        model.eval()
        y_true = []
        y_pred = []
        participant_ids = []
        
        with torch.no_grad():
            for batch in loader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                pids = batch['participant_id']
                
                predictions = model(features)
                
                # Denormalize predictions and labels if scaler exists
                if label_scaler is not None:
                    pred_denorm = label_scaler.inverse_transform(
                        predictions.cpu().numpy()
                    )
                    labels_denorm = label_scaler.inverse_transform(
                        labels.cpu().numpy()
                    )
                else:
                    pred_denorm = predictions.cpu().numpy()
                    labels_denorm = labels.cpu().numpy()
                
                y_true.extend(labels_denorm.flatten())
                y_pred.extend(pred_denorm.flatten())
                participant_ids.extend(pids)
        
        return np.array(y_true), np.array(y_pred), participant_ids
    
    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        if len(loader) == 0:
            logger.warning(f"No {split_name} data available.")
            continue
        
        logger.info(f"Evaluating {split_name} set...")
        y_true, y_pred, participant_ids = get_predictions_and_labels(
            model, loader, device, label_scaler
        )
        
        # Compute metrics
        window_metrics = evaluator.evaluate_window_level(y_true, y_pred)
        metrics_output[split_name] = window_metrics
        
        logger.info(f"{split_name.upper()} Metrics:")
        logger.info(f"  RMSE: {window_metrics['rmse']:.4f}")
        logger.info(f"  MAE:  {window_metrics['mae']:.4f}")
        logger.info(f"  R²:   {window_metrics['r2']:.4f}")
        
        corr = window_metrics.get('pearson_r') or window_metrics.get('pearson_corr') or window_metrics.get('correlation', 'N/A')
        if isinstance(corr, (int, float)):
            logger.info(f"  Corr: {corr:.4f}")
        else:
            logger.info(f"  Corr: {corr}")
        
        # Save predictions for test set
        if split_name == 'test':
            predictions_df = pd.DataFrame({
                'participant_id': participant_ids,
                'y_true': y_true,
                'y_pred': y_pred,
                'error': y_true - y_pred,
                'abs_error': np.abs(y_true - y_pred)
            })
            predictions_path = results_dir / 'predictions.csv'
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"✓ Saved predictions to {predictions_path}")
    
    # Save metrics
    metrics_path = results_dir / 'window_level_metrics.json'
    
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    metrics_output_serializable = convert_numpy_types(metrics_output)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output_serializable, f, indent=2)
    
    logger.info(f"✓ Saved metrics to {metrics_path}")
    
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETED!")
    logger.info("=" * 80)
    
    # Print summary
    if 'test' in metrics_output:
        test_metrics = metrics_output['test']
        logger.info("\nFINAL TEST SET RESULTS:")
        logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"  MAE:  {test_metrics['mae']:.4f}")
        logger.info(f"  R²:   {test_metrics['r2']:.4f}")
        
        corr = test_metrics.get('pearson_r') or test_metrics.get('pearson_corr') or test_metrics.get('correlation', 'N/A')
        if isinstance(corr, (int, float)):
            logger.info(f"  Correlation: {corr:.4f}")
        else:
            logger.info(f"  Correlation: {corr}")
        
        # Compare with baseline
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON WITH BASELINE:")
        logger.info("=" * 80)
        logger.info("Baseline (Original Model):")
        logger.info("  Test RMSE: 30.52")
        logger.info("  Test MAE:  22.51")
        logger.info("  Test R²:   -0.53")
        logger.info("\nImproved Model:")
        logger.info(f"  Test RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"  Test MAE:  {test_metrics['mae']:.4f}")
        logger.info(f"  Test R²:   {test_metrics['r2']:.4f}")
        
        rmse_improvement = ((30.52 - test_metrics['rmse']) / 30.52) * 100
        mae_improvement = ((22.51 - test_metrics['mae']) / 22.51) * 100
        r2_improvement = test_metrics['r2'] - (-0.53)
        
        logger.info("\nImprovements:")
        logger.info(f"  RMSE: {rmse_improvement:+.2f}%")
        logger.info(f"  MAE:  {mae_improvement:+.2f}%")
        logger.info(f"  R²:   {r2_improvement:+.4f}")

def main():
    """Main function"""
    base_dir = Path(__file__).parent.parent
    config = PipelineConfig(
        data_root=str(base_dir / "AI-READI"),
        output_dir=str(base_dir / "data" / "processed" / "results_improved_v2"),
        model_type="cnn",
        window_length_min=60,
        stride_min=60,
        batch_size=64,
        num_epochs=150,
        learning_rate=0.001,
        hidden_dim=256,  # Match training configuration
        num_layers=6,  # Match training configuration
        dropout=0.3,  # Match training configuration
        early_stopping_patience=25
    )
    
    results_dir = Path(config.output_dir)
    
    try:
        evaluate_improved_model(config, results_dir)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

