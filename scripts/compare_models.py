"""
Model Comparison Script
======================

Compares performance of CNN, LSTM, and Transformer models
with optimized configurations and longer windows.
"""

import sys
from pathlib import Path
import subprocess
import json
import pandas as pd

base_dir = Path(__file__).parent.parent

def run_training(model_type, window_hours):
    """Run training for a specific model and window length"""
    print(f"\n{'='*80}")
    print(f"Training {model_type.upper()} with {window_hours}h windows")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        str(base_dir / "scripts" / "train.py"),
        "--model", model_type,
        "--window_hours", str(window_hours)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR training {model_type} with {window_hours}h windows:")
        print(result.stderr)
        return None
    
    # Extract results
    output_dir = base_dir / "data" / "processed" / f"results_{model_type}_optimized_{window_hours}h"
    metrics_file = output_dir / "training_metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return {
            'model': model_type,
            'window_hours': window_hours,
            'best_val_loss': metrics['best_val_loss'],
            'best_val_mae': min(metrics['val_mae']),
            'best_val_rmse': min(metrics['val_rmse']),
            'best_epoch': metrics['best_epoch']
        }
    return None

def main():
    """Compare all models"""
    
    models = ['cnn', 'lstm', 'transformer']
    window_hours = [2, 3, 4]  # Test with 2, 3, and 4 hour windows
    
    results = []
    
    for model_type in models:
        for window_hours_val in window_hours:
            result = run_training(model_type, window_hours_val)
            if result:
                results.append(result)
    
    # Create comparison DataFrame
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(['model', 'window_hours'])
        
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        
        # Save results
        output_file = base_dir / "data" / "processed" / "model_comparison.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Find best model for each window length
        print("\n" + "="*80)
        print("BEST MODEL BY WINDOW LENGTH")
        print("="*80)
        for window_hours_val in window_hours:
            subset = df[df['window_hours'] == window_hours_val]
            best = subset.loc[subset['best_val_rmse'].idxmin()]
            print(f"\n{window_hours_val}h Windows:")
            print(f"  Best Model: {best['model'].upper()}")
            print(f"  Val RMSE: {best['best_val_rmse']:.4f}")
            print(f"  Val MAE: {best['best_val_mae']:.4f}")
    else:
        print("No results to compare!")

if __name__ == "__main__":
    main()

