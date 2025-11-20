"""
Improved 1D CNN Training V2 - Better Accuracy
==============================================
Fixes:
1. Increased epochs and early stopping patience
2. Better model initialization (output bias initialization)
3. Reduced dropout for better learning
4. Better learning rate schedule
5. Increased model capacity
6. Better regularization balance
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
        logging.FileHandler(log_dir / 'training.log')
    ]
)
logger = logging.getLogger(__name__)

class ImprovedStressTrainerV2:
    """Improved trainer with better initialization and training strategy"""
    
    def __init__(self, model, config, output_dir, label_scaler=None):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.label_scaler = label_scaler
        
        # Improved optimizer with moderate weight decay
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=5e-4  # Moderate weight decay
        )
        
        # Learning rate scheduler - more aggressive reduction
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7
        )
        
        # Huber loss - less sensitive to outliers, better for regression
        # Combines benefits of MSE (for small errors) and MAE (for large errors)
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
            def __init__(self, patience=25):
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
    """Apply label scaling to windows (creates new windows to avoid modifying original)"""
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
            # Xavier uniform initialization
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                # Initialize bias to zero (labels are normalized to mean=0)
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            # Kaiming uniform for ReLU
            torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Initialize output layer bias to match normalized label distribution
    # Since labels are normalized to mean=0, std=1, initialize bias close to 0
    if hasattr(model, 'fc2') and model.fc2.bias is not None:
        # Initialize final output bias to 0 (labels normalized to mean=0)
        torch.nn.init.zeros_(model.fc2.bias)
        logger.info("✓ Output layer bias initialized to 0 (labels normalized)")
    elif hasattr(model, 'fc') and model.fc.bias is not None:
        # Fallback for older model architecture
        torch.nn.init.zeros_(model.fc.bias)
        logger.info("✓ Output layer bias initialized to 0 (labels normalized)")

def main():
    """Run improved training V2"""
    
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("IMPROVED 1D CNN TRAINING V2 - BETTER ACCURACY")
    logger.info("=" * 80)
    
    # Improved configuration
    base_dir = Path(__file__).parent.parent
    improved_output_dir = str(base_dir / "data" / "processed" / "results_improved_v2")
    
    # FIXED cache directory - always use the same location
    fixed_cache_dir = str(base_dir / "data" / "processed" / "preprocessed_cache")
    
    config = PipelineConfig(
        data_root=str(base_dir / "AI-READI"),
        output_dir=improved_output_dir,
        
        model_type="cnn",
        window_length_min=60,
        stride_min=60,
        
        # Improved hyperparameters for better accuracy (reduced RMSE/MAE)
        batch_size=64,  # Smaller batch for better gradient estimates
        num_epochs=300,  # More epochs for better convergence
        early_stopping_patience=30,  # More patience
        learning_rate=0.001,  # Higher initial LR for faster learning
        
        # Increased model capacity for better accuracy
        hidden_dim=256,  # Larger capacity for better learning
        num_layers=6,  # More layers for better feature extraction
        dropout=0.3,  # Reduced dropout to allow more learning
    )
    
    # Set the fixed cache directory in config
    config.cache_dir = fixed_cache_dir
    
    # Feature selection based on importance analysis
    config.selected_modalities = ['heart_rate', 'cgm', 'respiratory_rate']
    config.enable_heart_rate_engineering = True
    
    # Check cache status
    cache_path = Path(fixed_cache_dir)
    if cache_path.exists():
        cache_count = len(list(cache_path.glob("*.pkl")))
        logger.info(f"✓ Using fixed cache directory: {fixed_cache_dir}")
        logger.info(f"✓ Cache contains {cache_count}/903 files")
        logger.info(f"  Missing files will be preprocessed and saved to cache")
    else:
        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created cache directory: {fixed_cache_dir}")
        logger.info(f"  All participants will be preprocessed and cached")
    
    # Create output directory (for models, logs, etc. - NOT cache)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "models").mkdir(exist_ok=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("IMPROVED CONFIGURATION V2")
    logger.info("=" * 80)
    logger.info(f"Dropout: {config.dropout} (reduced for better learning)")
    logger.info(f"Batch Size: {config.batch_size} (optimized for accuracy)")
    logger.info(f"Learning Rate: {config.learning_rate} (higher for faster learning)")
    logger.info(f"Weight Decay: 5e-4 (moderate regularization)")
    logger.info(f"Loss: Huber Loss (better for regression, less sensitive to outliers)")
    logger.info(f"Hidden Dim: {config.hidden_dim} (increased capacity)")
    logger.info(f"Num Layers: {config.num_layers} (more layers for better features)")
    logger.info(f"Epochs: {config.num_epochs} (with early stopping)")
    logger.info(f"Early Stopping Patience: {config.early_stopping_patience}")
    logger.info(f"Label Normalization: Enabled")
    logger.info(f"Selected Modalities: {config.selected_modalities}")
    logger.info(f"HR Feature Engineering: {config.enable_heart_rate_engineering}")
    logger.info(f"Gradient Clipping: 0.5 (tighter for stability)")
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
    
    # Check if we have any windows
    if len(all_windows) == 0:
        logger.error("No windows created! This could be due to:")
        logger.error("  1. All windows filtered out (30% valid data threshold too strict)")
        logger.error("  2. No stress labels found in windows")
        logger.error("  3. Window creation failed silently")
        logger.error("Please check the windowing logic and data quality.")
        return 1
    
    # Step 2: Create splits
    logger.info("Step 2: Creating data splits...")
    participants_df = data_ingestion.participants_df
    train_windows, val_windows, test_windows = data_splitter.create_splits(
        all_windows, participants_df
    )
    
    logger.info(f"Train: {len(train_windows)}, Val: {len(val_windows)}, Test: {len(test_windows)}")
    
    # Check if we have training data
    if len(train_windows) == 0:
        logger.error("No training windows available! Cannot proceed with training.")
        return 1
    
    # Step 3: Create label scaler and apply scaling
    logger.info("Step 3: Normalizing stress labels...")
    label_scaler = create_label_scaler(train_windows)
    
    if label_scaler is not None:
        train_windows = apply_label_scaling(train_windows, label_scaler)
        val_windows = apply_label_scaling(val_windows, label_scaler)
        test_windows = apply_label_scaling(test_windows, label_scaler)
        logger.info("✓ Labels normalized")
    else:
        logger.warning("⚠ Label scaler not created, using raw labels")
    
    # Step 4: Create data loaders
    logger.info("Step 4: Creating data loaders...")
    train_loader, val_loader, test_loader = data_splitter.create_dataloaders(
        train_windows, val_windows, test_windows
    )
    
    # Step 5: Create model
    logger.info("Step 5: Creating model...")
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['features'].shape[2]
    
    model = ModelFactory.create_model(config.model_type, input_dim, config)
    
    # Better initialization
    initialize_model_weights(model, label_scaler)
    logger.info("✓ Model created and initialized")
    
    # Step 6: Train model
    logger.info("Step 6: Training model...")
    
    trainer = ImprovedStressTrainerV2(model, config, config.output_dir, label_scaler)
    
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
            features = batch['features'].to(trainer.device)
            labels = batch['label'].to(trainer.device)
            
            # Labels are already normalized in windows
            labels_norm = labels
            
            # Forward pass
            trainer.optimizer.zero_grad()
            predictions = model(features)
            
            # Compute loss
            loss = trainer.criterion(predictions, labels_norm)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # Less aggressive clipping for larger model
            
            trainer.optimizer.step()
            
            # Denormalize for metrics
            if label_scaler is not None:
                # Ensure predictions are 2D for inverse_transform
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
                features = batch['features'].to(trainer.device)
                labels = batch['label'].to(trainer.device)
                
                labels_norm = labels
                
                predictions = model(features)
                loss = trainer.criterion(predictions, labels_norm)
                
                # Denormalize for metrics
                if label_scaler is not None:
                    # Ensure predictions are 2D for inverse_transform
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

