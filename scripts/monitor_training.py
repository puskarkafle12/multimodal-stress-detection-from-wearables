#!/usr/bin/env python3
"""
Monitor Training Progress
========================
Shows real-time training progress by monitoring the training_metrics.json file
"""

import sys
import time
import json
from pathlib import Path

def monitor_training(results_dir):
    """Monitor training progress"""
    results_path = Path(results_dir)
    metrics_file = results_path / "training_metrics.json"
    
    print("=" * 80)
    print("MONITORING LSTM TRAINING PROGRESS")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Metrics file: {metrics_file}")
    print("=" * 80)
    print("\nWaiting for training to start...")
    print("(Press Ctrl+C to stop monitoring)\n")
    
    last_epoch = 0
    
    while True:
        try:
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                train_losses = metrics.get('train_loss', [])
                val_losses = metrics.get('val_loss', [])
                val_maes = metrics.get('val_mae', [])
                val_rmses = metrics.get('val_rmse', [])
                
                current_epoch = len(train_losses)
                
                if current_epoch > last_epoch:
                    # New epoch completed
                    print(f"\n{'='*80}")
                    print(f"EPOCH {current_epoch}")
                    print(f"{'='*80}")
                    
                    if train_losses:
                        print(f"Train Loss: {train_losses[-1]:.4f}")
                    
                    if val_losses:
                        print(f"Val Loss:   {val_losses[-1]:.4f}")
                        print(f"Val MAE:    {val_maes[-1]:.4f}" if val_maes else "")
                        print(f"Val RMSE:   {val_rmses[-1]:.4f}" if val_rmses else "")
                    
                    if metrics.get('best_epoch'):
                        print(f"\nBest Epoch: {metrics['best_epoch']}")
                        print(f"Best Val Loss: {metrics.get('best_val_loss', 'N/A'):.4f}")
                        if val_rmses:
                            print(f"Best Val RMSE: {min(val_rmses):.4f}")
                        if val_maes:
                            print(f"Best Val MAE: {min(val_maes):.4f}")
                    
                    last_epoch = current_epoch
                else:
                    # Still processing same epoch
                    print(".", end="", flush=True)
            else:
                # Data loading phase
                print(".", end="", flush=True)
            
            time.sleep(2)  # Check every 2 seconds
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(2)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "data/processed/results_lstm_optimized_2h"
    
    monitor_training(results_dir)

