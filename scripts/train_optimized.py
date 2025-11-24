"""
Optimized Training Script - Architecture-Specific Configurations
================================================================

This script trains models with optimized hyperparameters for each architecture:
- Longer windows (2-4 hours) for better temporal modeling
- Architecture-specific hyperparameters
- Better utilization of each model's strengths
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import PipelineConfig
import logging
import json
import time
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np

# Setup logging
base_dir = Path(__file__).parent.parent
log_dir = base_dir / "data" / "processed" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / 'training_optimized.log')
    ]
)
logger = logging.getLogger(__name__)

class ImprovedStressTrainerV2:
    """Improved trainer with better initialization and training strategy"""
    
    def __init__(self, model, config, output_dir, label_scaler=None, weight_decay=5e-4):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.label_scaler = label_scaler
        
        # Improved optimizer with moderate weight decay
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler - more aggressive reduction
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7
        )
        
        # Huber loss - less sensitive to outliers, better for regression
        self.criterion = nn.HuberLoss(delta=1.0)
        
        # Early stopping with more patience
        self.early_stopping = self._create_early_stopping(config.early_stopping_patience)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Improved trainer V2 initialized on {self.device}")
        logger.info(f"Using Huber loss (delta=1.0), weight decay: 5e-4")
    
    def _create_early_stopping(self, patience):
        """Create early stopping callback"""
        class EarlyStopping:
            def __init__(self, patience=30):  # Increased patience for early fusion
                self.patience = patience
                self.best_loss = float('inf')
                self.counter = 0
                self.best_weights = None
            
            def __call__(self, val_loss, model):
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.counter = 0
                    self.best_weights = model.state_dict().copy()
                else:
                    self.counter += 1
                
                if self.counter >= self.patience:
                    if self.best_weights is not None:
                        model.load_state_dict(self.best_weights)
                    return True
                return False
        
        return EarlyStopping(patience)

def create_label_scaler(train_windows):
    """Create label scaler from training windows"""
    from sklearn.preprocessing import StandardScaler
    
    train_labels = [w['stress_label'] for w in train_windows if w.get('stress_label') is not None]
    
    if not train_labels:
        return None
    
    scaler = StandardScaler()
    scaler.fit(np.array(train_labels).reshape(-1, 1))
    
    logger.info(f"Label scaler fitted: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")
    
    return scaler

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

def initialize_model_weights(model, label_scaler=None):
    """Initialize model weights, especially output layer bias"""
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Initialize output layer bias
    if hasattr(model, 'fc2') and model.fc2.bias is not None:
        torch.nn.init.zeros_(model.fc2.bias)
        logger.info("✓ Output layer bias initialized to 0 (labels normalized)")
    elif hasattr(model, 'fc') and model.fc.bias is not None:
        torch.nn.init.zeros_(model.fc.bias)
        logger.info("✓ Output layer bias initialized to 0 (labels normalized)")
    
    # Initialize CLS token for transformer models
    if hasattr(model, 'cls_token'):
        torch.nn.init.normal_(model.cls_token, mean=0.0, std=0.02)
        logger.info("✓ CLS token initialized for transformer")

def get_model_config(model_type, window_length_min):
    """Get optimized configuration for each model type"""
    
    base_config = {
        'window_length_min': window_length_min,
        'stride_min': window_length_min,  # Non-overlapping windows
        'batch_size': 64,
        'num_epochs': 300,
        'early_stopping_patience': 30,
        'learning_rate': 0.001,
        'dropout': 0.4,  # Increased for better regularization
        'weight_decay': 1e-3,  # Increased weight decay for regularization
    }
    
    if model_type == "cnn":
        return {
            **base_config,
            'hidden_dim': 256,
            'num_layers': 6,
            'learning_rate': 0.001,  # CNN works well with standard LR
        }
    elif model_type == "lstm":
        return {
            **base_config,
            'hidden_dim': 256,
            'num_layers': 4,  # Fewer layers for LSTM (gradient flow)
            'learning_rate': 0.0003,  # Lower LR to prevent overfitting (was 0.0008)
            'dropout': 0.4,  # More dropout for regularization
            'bidirectional': True,  # Enable bidirectional for better temporal modeling
        }
    elif model_type == "transformer":
        return {
            **base_config,
            'hidden_dim': 256,
            'num_layers': 4,  # Fewer layers but deeper attention
            'num_heads': 8,  # Multi-head attention
            'learning_rate': 0.0005,  # Lower LR for transformer stability
            'dropout': 0.2,  # Less dropout for transformers
        }
    elif model_type == "early_fusion":
        return {
            **base_config,
            'hidden_dim': 256,
            'num_layers': 4,
            'learning_rate': 0.0003,  # Lower LR to prevent overfitting (was 0.0008)
            'dropout': 0.5,  # Higher dropout for better regularization (was 0.3)
            'weight_decay': 1e-3,  # Increased weight decay (was 5e-4)
            'encoder_type': 'cnn',  # Default encoder
        }
    elif model_type == "late_fusion":
        return {
            **base_config,
            'hidden_dim': 128,  # Per-modality encoder hidden dim
            'num_layers': 2,  # Fewer layers per modality encoder
            'learning_rate': 0.0005,  # Lower LR for attention fusion
            'dropout': 0.3,
            'fusion_type': 'attention',  # Use attention fusion
        }
    else:
        return base_config

def main():
    """Run optimized training"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Train optimized model')
    parser.add_argument('--model', type=str, default='cnn', 
                       choices=['cnn', 'lstm', 'transformer', 'early_fusion', 'late_fusion'],
                       help='Model type to train')
    parser.add_argument('--encoder_type', type=str, default='lstm',
                       choices=['cnn', 'lstm', 'transformer'],
                       help='Encoder type for early_fusion model (only used when --model=early_fusion)')
    parser.add_argument('--window_hours', type=int, default=2,
                       choices=[1, 2, 3, 4],
                       help='Window length in hours')
    parser.add_argument('--max_participants', type=int, default=None,
                       help='Maximum number of participants to process (for quick testing)')
    args = parser.parse_args()
    
    start_time = time.time()
    
    model_type = args.model
    window_length_min = args.window_hours * 60
    
    logger.info("=" * 80)
    if model_type == "early_fusion":
        logger.info(f"EARLY FUSION TRAINING ({args.encoder_type.upper()} encoder) - {args.window_hours}H WINDOWS")
    elif model_type == "late_fusion":
        logger.info(f"LATE FUSION TRAINING (Attention-based) - {args.window_hours}H WINDOWS")
    else:
        logger.info(f"OPTIMIZED {model_type.upper()} TRAINING - {args.window_hours}H WINDOWS")
    logger.info("=" * 80)
    
    # Get optimized configuration
    if model_type == "early_fusion":
        # Use encoder_type config for early fusion
        model_config = get_model_config(args.encoder_type, window_length_min)
    elif model_type == "late_fusion":
        # Late fusion has its own config
        model_config = get_model_config("late_fusion", window_length_min)
    else:
        model_config = get_model_config(model_type, window_length_min)
    
    # Improved configuration
    base_dir = Path(__file__).parent.parent
    if model_type == "early_fusion":
        output_dir_name = f"results_early_fusion_{args.encoder_type}_optimized_{args.window_hours}h"
    else:
        output_dir_name = f"results_{model_type}_optimized_{args.window_hours}h"
    improved_output_dir = str(base_dir / "data" / "processed" / output_dir_name)
    
    # Model-specific cache directory - each model has its own cache
    fixed_cache_dir = str(base_dir / "data" / "processed" / f"preprocessed_cache_{model_type}")
    
    config = PipelineConfig(
        data_root=str(base_dir / "AI-READI"),
        output_dir=improved_output_dir,
        model_type=model_type,
        **{k: v for k, v in model_config.items() if k not in ['bidirectional', 'num_heads', 'encoder_type', 'weight_decay', 'fusion_type']}
    )
    
    # Get weight_decay separately (not a PipelineConfig field)
    weight_decay = model_config.get('weight_decay', 5e-4)
    
    # Set encoder-specific parameters
    if model_type == "early_fusion":
        config.encoder_type = args.encoder_type
        if args.encoder_type == "lstm" and 'bidirectional' in model_config:
            config.bidirectional = model_config['bidirectional']
        if args.encoder_type == "transformer" and 'num_heads' in model_config:
            config.num_heads = model_config['num_heads']
    elif model_type == "lstm" and 'bidirectional' in model_config:
        config.bidirectional = model_config['bidirectional']
    elif model_type == "transformer" and 'num_heads' in model_config:
        config.num_heads = model_config['num_heads']
    
    # Set the fixed cache directory in config
    config.cache_dir = fixed_cache_dir
    
    # Feature selection based on importance analysis
    config.selected_modalities = ['heart_rate', 'cgm', 'respiratory_rate']
    config.enable_heart_rate_engineering = True
    
    # Check cache status
    cache_path = Path(fixed_cache_dir)
    if cache_path.exists():
        cache_count = len(list(cache_path.glob("*.pkl")))
        logger.info(f"✓ Using model-specific cache directory: {fixed_cache_dir}")
        logger.info(f"✓ Cache contains {cache_count}/903 files")
        logger.info(f"  Missing files will be preprocessed and saved to this cache")
    else:
        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created model-specific cache directory: {fixed_cache_dir}")
        logger.info(f"  All participants will be preprocessed and cached here")
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "models").mkdir(exist_ok=True)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"OPTIMIZED {model_type.upper()} CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model Type: {model_type.upper()}")
    if model_type == "early_fusion":
        logger.info(f"Encoder Type: {config.encoder_type.upper()}")
    logger.info(f"Window Length: {window_length_min} minutes ({args.window_hours} hours)")
    logger.info(f"Window Samples: {int(window_length_min / 5)} (at 5-min intervals)")
    logger.info(f"Dropout: {config.dropout}")
    logger.info(f"Batch Size: {config.batch_size}")
    logger.info(f"Learning Rate: {config.learning_rate}")
    logger.info(f"Weight Decay: {config.weight_decay if hasattr(config, 'weight_decay') else 5e-4}")
    logger.info(f"Loss: Huber Loss")
    logger.info(f"Hidden Dim: {config.hidden_dim}")
    logger.info(f"Num Layers: {config.num_layers}")
    if model_type == "early_fusion" and config.encoder_type == "lstm" and hasattr(config, 'bidirectional'):
        logger.info(f"Bidirectional: {config.bidirectional}")
    elif model_type == "lstm" and hasattr(config, 'bidirectional'):
        logger.info(f"Bidirectional: {config.bidirectional}")
    if model_type == "early_fusion" and config.encoder_type == "transformer" and hasattr(config, 'num_heads'):
        logger.info(f"Num Heads: {config.num_heads}")
    elif model_type == "transformer" and hasattr(config, 'num_heads'):
        logger.info(f"Num Heads: {config.num_heads}")
    logger.info(f"Epochs: {config.num_epochs} (with early stopping)")
    logger.info(f"Early Stopping Patience: {config.early_stopping_patience}")
    logger.info(f"Selected Modalities: {config.selected_modalities}")
    logger.info(f"HR Feature Engineering: {config.enable_heart_rate_engineering}")
    logger.info("=" * 80 + "\n")
    
    # Initialize pipeline components
    from data_loader import EnhancedPreprocessingDataIngestion
    from windowing import WindowingSystem
    from data_splits import DataSplitter
    from models import ModelFactory
    
    data_ingestion = EnhancedPreprocessingDataIngestion(config)
    windowing_system = WindowingSystem(config)
    data_splitter = DataSplitter(config)
    
    # Step 1: Load data
    logger.info("Step 1: Loading data...")
    all_windows = []
    participants = data_ingestion.get_available_participants()
    
    # Limit participants if specified
    if args.max_participants is not None:
        participants = participants[:args.max_participants]
        logger.info(f"⚠️  LIMITED TO FIRST {args.max_participants} PARTICIPANTS (QUICK TEST MODE)")
    
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
    
    if len(all_windows) == 0:
        logger.error("No windows created!")
        return 1
    
    # Step 2: Create splits
    logger.info("Step 2: Creating data splits...")
    participants_df = data_ingestion.participants_df
    train_windows, val_windows, test_windows = data_splitter.create_splits(
        all_windows, participants_df
    )
    
    logger.info(f"Train: {len(train_windows)}, Val: {len(val_windows)}, Test: {len(test_windows)}")
    
    if len(train_windows) == 0:
        logger.error("No training windows available!")
        return 1
    
    # Step 3: Create label scaler and apply scaling
    logger.info("Step 3: Normalizing stress labels...")
    label_scaler = create_label_scaler(train_windows)
    
    if label_scaler is not None:
        train_windows = apply_label_scaling(train_windows, label_scaler)
        val_windows = apply_label_scaling(val_windows, label_scaler)
        test_windows = apply_label_scaling(test_windows, label_scaler)
        logger.info("✓ Labels normalized")
    
    # Step 4: Create data loaders
    logger.info("Step 4: Creating data loaders...")
    
    # For late fusion, need custom collate function
    if model_type == "late_fusion":
        from late_fusion_adapter import LateFusionDatasetAdapter, late_fusion_collate_fn
        from torch.utils.data import DataLoader
        
        # Create datasets
        from data_splits import StressDataset
        train_dataset = StressDataset(train_windows, fit_scaler=True)
        val_dataset = StressDataset(val_windows, scaler=train_dataset.scaler)
        test_dataset = StressDataset(test_windows, scaler=train_dataset.scaler)
        
        # Wrap with adapter
        logger.info("Converting data format for late fusion (concatenated -> dict)...")
        train_dataset = LateFusionDatasetAdapter(train_dataset)
        val_dataset = LateFusionDatasetAdapter(val_dataset)
        test_dataset = LateFusionDatasetAdapter(test_dataset)
        
        # Create loaders with custom collate function
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                                 collate_fn=late_fusion_collate_fn, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                               collate_fn=late_fusion_collate_fn, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                collate_fn=late_fusion_collate_fn, num_workers=0, pin_memory=True)
    else:
        train_loader, val_loader, test_loader = data_splitter.create_dataloaders(
            train_windows, val_windows, test_windows
        )
    
    # Step 5: Create model
    logger.info("Step 5: Creating model...")
    # For late fusion, input_dim is per-modality (2), not total
    if model_type == "late_fusion":
        input_dim = 2  # Each modality has 2 channels (value + mask)
    else:
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch['features'].shape[2]
    
    # Create model with optimized parameters
    if model_type == "lstm" and hasattr(config, 'bidirectional'):
        from models import StressLSTM
        model = StressLSTM(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional
        )
    elif model_type == "transformer" and hasattr(config, 'num_heads'):
        from models import StressTransformer
        model = StressTransformer(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            window_length=int(config.window_length_min / 5)
        )
    else:
        model = ModelFactory.create_model(config.model_type, input_dim, config)
    
    # Better initialization
    initialize_model_weights(model, label_scaler)
    logger.info("✓ Model created and initialized")
    
    # Step 6: Train model
    logger.info("Step 6: Training model...")
    
    trainer = ImprovedStressTrainerV2(model, config, config.output_dir, label_scaler, weight_decay=weight_decay)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_maes = []
    val_rmses = []
    
    for epoch in range(config.num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Handle late fusion (dict) vs early fusion (tensor)
            if isinstance(batch['features'], dict):
                features = {k: v.to(trainer.device) for k, v in batch['features'].items()}
            else:
                features = batch['features'].to(trainer.device)
            labels = batch['label'].to(trainer.device)
            
            labels_norm = labels
            
            # Forward pass
            trainer.optimizer.zero_grad()
            predictions = model(features)
            
            # Compute loss
            loss = trainer.criterion(predictions, labels_norm)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            trainer.optimizer.step()
            
            # Denormalize for metrics
            if label_scaler is not None:
                pred_np = predictions.detach().cpu().numpy()
                if pred_np.ndim == 1:
                    pred_np = pred_np.reshape(-1, 1)
                elif pred_np.shape[1] != 1:
                    pred_np = pred_np.flatten().reshape(-1, 1)
                
                labels_np = labels_norm.cpu().numpy()
                if labels_np.ndim == 1:
                    labels_np = labels_np.reshape(-1, 1)
                elif labels_np.shape[1] != 1:
                    labels_np = labels_np.flatten().reshape(-1, 1)
                
                pred_denorm = torch.FloatTensor(label_scaler.inverse_transform(pred_np)).squeeze()
                labels_denorm = torch.FloatTensor(label_scaler.inverse_transform(labels_np)).squeeze()
            else:
                pred_denorm = predictions.detach().cpu().squeeze()
                labels_denorm = labels.cpu().squeeze()
            
            mae = torch.mean(torch.abs(pred_denorm - labels_denorm)).item()
            
            train_loss += loss.item()
            train_mae += mae
            num_batches += 1
        
        train_loss /= num_batches
        train_mae /= num_batches
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_rmse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle late fusion (dict) vs early fusion (tensor)
                if isinstance(batch['features'], dict):
                    features = {k: v.to(trainer.device) for k, v in batch['features'].items()}
                else:
                    features = batch['features'].to(trainer.device)
                labels = batch['label'].to(trainer.device)
                
                labels_norm = labels
                
                predictions = model(features)
                loss = trainer.criterion(predictions, labels_norm)
                
                # Denormalize for metrics
                if label_scaler is not None:
                    pred_np = predictions.cpu().numpy()
                    if pred_np.ndim == 1:
                        pred_np = pred_np.reshape(-1, 1)
                    elif pred_np.shape[1] != 1:
                        pred_np = pred_np.flatten().reshape(-1, 1)
                    
                    labels_np = labels_norm.cpu().numpy()
                    if labels_np.ndim == 1:
                        labels_np = labels_np.reshape(-1, 1)
                    elif labels_np.shape[1] != 1:
                        labels_np = labels_np.flatten().reshape(-1, 1)
                    
                    pred_denorm = torch.FloatTensor(label_scaler.inverse_transform(pred_np)).squeeze()
                    labels_denorm = torch.FloatTensor(label_scaler.inverse_transform(labels_np)).squeeze()
                else:
                    pred_denorm = predictions.cpu().squeeze()
                    labels_denorm = labels.cpu().squeeze()
                
                mae = torch.mean(torch.abs(pred_denorm - labels_denorm)).item()
                rmse = torch.sqrt(torch.mean((pred_denorm - labels_denorm) ** 2)).item()
                
                val_loss += loss.item()
                val_mae += mae
                val_rmse += rmse
                num_batches += 1
        
        val_loss /= num_batches
        val_mae /= num_batches
        val_rmse /= num_batches
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_rmses.append(val_rmse)
        
        # Update learning rate
        trainer.scheduler.step(val_loss)
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{config.num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}, "
            f"LR: {trainer.optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Early stopping
        if trainer.early_stopping(val_loss, model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': config,
                'label_scaler': label_scaler,
                'epoch': epoch + 1,
                'val_loss': val_loss
            }, config.output_dir / "models" / "best_model.pt")
            logger.info(f"✓ Saved best model (Val Loss: {val_loss:.4f})")
    
    # Save training metrics
    training_metrics = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_mae': val_maes,
        'val_rmse': val_rmses,
        'best_epoch': len(val_losses) - trainer.early_stopping.counter,
        'best_val_loss': best_val_loss
    }
    
    with open(config.output_dir / "training_metrics.json", 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Best Val MAE: {min(val_maes):.4f}")
    logger.info(f"Best Val RMSE: {min(val_rmses):.4f}")
    logger.info(f"Model saved to: {config.output_dir / 'models' / 'best_model.pt'}")
    
    # Save label scaler
    if label_scaler is not None:
        import pickle
        with open(config.output_dir / "label_scaler.pkl", 'wb') as f:
            pickle.dump(label_scaler, f)
        logger.info(f"Label scaler saved to: {config.output_dir / 'label_scaler.pkl'}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total time: {elapsed_time/60:.2f} minutes")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

