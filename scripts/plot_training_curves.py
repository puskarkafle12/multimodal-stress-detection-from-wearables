#!/usr/bin/env python3
"""
Plot training and validation loss curves from training metrics.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys


def load_training_metrics(metrics_path):
    """Load training metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def plot_training_curves(metrics, output_path=None, show_plot=True):
    """
    Plot training and validation loss curves.
    
    Args:
        metrics: Dictionary containing training metrics
        output_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    train_losses = metrics.get('train_loss', [])
    val_losses = metrics.get('val_loss', [])
    val_maes = metrics.get('val_mae', [])
    val_rmses = metrics.get('val_rmse', [])
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=1)
    
    # Highlight best epoch if available
    if 'best_epoch' in metrics and metrics['best_epoch'] > 0:
        best_epoch = metrics['best_epoch']
        if best_epoch <= len(val_losses):
            best_val_loss = val_losses[best_epoch - 1]
            ax1.plot(best_epoch, best_val_loss, 'go', markersize=10, label=f'Best (Epoch {best_epoch})')
            ax1.legend(fontsize=10)
    
    # Plot 2: Validation Loss (zoomed)
    ax2 = axes[0, 1]
    ax2.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss (Detailed)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=1)
    
    if 'best_epoch' in metrics and metrics['best_epoch'] > 0:
        best_epoch = metrics['best_epoch']
        if best_epoch <= len(val_losses):
            best_val_loss = val_losses[best_epoch - 1]
            ax2.plot(best_epoch, best_val_loss, 'go', markersize=10, label=f'Best (Epoch {best_epoch})')
            ax2.legend(fontsize=10)
    
    # Plot 3: Validation MAE
    if val_maes:
        ax3 = axes[1, 0]
        ax3.plot(epochs, val_maes, 'g-', label='Val MAE', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Mean Absolute Error', fontsize=12)
        ax3.set_title('Validation MAE', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(left=1)
        
        if 'best_epoch' in metrics and metrics['best_epoch'] > 0:
            best_epoch = metrics['best_epoch']
            if best_epoch <= len(val_maes):
                best_mae = val_maes[best_epoch - 1]
                ax3.plot(best_epoch, best_mae, 'ro', markersize=10, label=f'Best (Epoch {best_epoch})')
                ax3.legend(fontsize=10)
    
    # Plot 4: Validation RMSE
    if val_rmses:
        ax4 = axes[1, 1]
        ax4.plot(epochs, val_rmses, 'm-', label='Val RMSE', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Root Mean Squared Error', fontsize=12)
        ax4.set_title('Validation RMSE', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(left=1)
        
        if 'best_epoch' in metrics and metrics['best_epoch'] > 0:
            best_epoch = metrics['best_epoch']
            if best_epoch <= len(val_rmses):
                best_rmse = val_rmses[best_epoch - 1]
                ax4.plot(best_epoch, best_rmse, 'ro', markersize=10, label=f'Best (Epoch {best_epoch})')
                ax4.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training and validation loss curves')
    parser.add_argument(
        '--metrics-file',
        type=str,
        default=None,
        help='Path to training_metrics.json file (default: auto-detect from results_improved_v2)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the plot (default: save in same directory as metrics)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot (only save)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect metrics file if not provided
    if args.metrics_file is None:
        default_path = Path(__file__).parent.parent / "data" / "processed" / "results_improved_v2" / "training_metrics.json"
        if default_path.exists():
            args.metrics_file = str(default_path)
            print(f"Using default metrics file: {args.metrics_file}")
        else:
            print(f"Error: Could not find training_metrics.json at default location: {default_path}")
            print("Please specify --metrics-file path")
            sys.exit(1)
    
    # Load metrics
    metrics_path = Path(args.metrics_file)
    if not metrics_path.exists():
        print(f"Error: Metrics file not found: {metrics_path}")
        sys.exit(1)
    
    print(f"Loading metrics from: {metrics_path}")
    metrics = load_training_metrics(metrics_path)
    
    # Print summary
    print("\n" + "="*60)
    print("Training Metrics Summary")
    print("="*60)
    print(f"Total Epochs: {len(metrics.get('train_loss', []))}")
    print(f"Best Epoch: {metrics.get('best_epoch', 'N/A')}")
    print(f"Best Val Loss: {metrics.get('best_val_loss', 'N/A'):.4f}" if metrics.get('best_val_loss') else "Best Val Loss: N/A")
    if metrics.get('val_mae'):
        print(f"Best Val MAE: {min(metrics['val_mae']):.4f}")
    if metrics.get('val_rmse'):
        print(f"Best Val RMSE: {min(metrics['val_rmse']):.4f}")
    print("="*60 + "\n")
    
    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = metrics_path.parent / "training_curves.png"
    
    # Plot curves
    plot_training_curves(metrics, output_path=output_path, show_plot=not args.no_show)


if __name__ == "__main__":
    main()

