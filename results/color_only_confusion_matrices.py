#!/usr/bin/env python3
"""
Color-Only Confusion Matrix Analysis for Blur Predictions
Generates confusion matrices for color classification only (no vehicle types)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from pathlib import Path
import argparse

def load_prediction_data(blur_level):
    """Load prediction data for a specific blur level."""
    # Look for directories with the pattern blur_{level}_run_*
    blur_dirs = list(Path(".").glob(f"blur_{blur_level}_run_*"))
    
    if not blur_dirs:
        print(f"No blur directories found for blur level {blur_level}")
        return None
    
    all_predictions = []
    
    for blur_dir in blur_dirs:
        json_file = blur_dir / "predictions_with_scores.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    all_predictions.extend(data)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {json_file}: {e}")
                continue
        else:
            print(f"No predictions file found in {blur_dir}")
    
    return all_predictions

def create_color_confusion_matrix(blur_level):
    """Create confusion matrix for color classification only."""
    # Load predictions
    predictions = load_prediction_data(blur_level)
    if not predictions:
        return None
    
    # Map color labels to indices
    color_mapping = {
        'color_white': 0,
        'color_silver/grey': 1,
        'color_black': 2,
        'color_blue': 3,
        'color_other': 4,
        'color_red': 5,
        'color_yellow': 6,
        'color_green': 7
    }
    
    # Reference labels (8 colors)
    labels = [
        'white',
        'silver/grey',
        'black',
        'blue',
        'other',
        'red',
        'yellow',
        'green'
    ]
    
    y_true = []
    y_pred = []
    
    for pred in predictions:
        # Extract true color
        true_labels = pred.get('true', [])
        true_color = None
        for label in true_labels:
            if label in color_mapping:
                true_color = color_mapping[label]
                break
        
        # Extract predicted color (find highest scoring color)
        pred_scores = pred.get('pred_scores', {})
        best_color = None
        best_score = -1
        
        for color, idx in color_mapping.items():
            score = pred_scores.get(color, 0)
            if score > best_score:
                best_score = score
                best_color = idx
        
        if true_color is not None and best_color is not None:
            y_true.append(true_color)
            y_pred.append(best_color)
    
    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
        return {
            'matrix': cm,
            'count': len(y_true),
            'labels': labels
        }
    
    return None

def plot_color_confusion_matrix(blur_level, matrix_data, sigma_value):
    """Plot confusion matrix for color classification."""
    if not matrix_data:
        print(f"No matrix data to plot for blur level {blur_level}")
        return
    
    cm = matrix_data['matrix']
    count = matrix_data['count']
    labels = matrix_data['labels']
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlBu_r', 
               xticklabels=labels, yticklabels=labels,
               ax=ax, cbar_kws={'shrink': 0.8})
    
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    ax.set_title(f'Color Classification Confusion Matrix - Blur σ = {sigma_value:.2f}\n({count} samples, accuracy: {accuracy:.3f})', 
                fontweight='bold', fontsize=14)
    ax.set_xlabel('Predicted Color', fontsize=12)
    ax.set_ylabel('True Color', fontsize=12)
    
    # Rotate labels
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', rotation=0, labelsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"confusion_matrix_color_only_blur_{blur_level}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved color-only confusion matrix for blur level {blur_level} to {output_file}")
    plt.close()

def plot_color_grid(all_matrices_data):
    """Plot grid of color confusion matrices across blur levels."""
    if not all_matrices_data:
        print("No matrix data to plot")
        return
    
    sigma_values = np.linspace(0.0, 4.0, 12)
    
    # Create grid layout
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(24, 18))
    fig.suptitle('Color Classification Confusion Matrices Across Blur Levels', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for blur_level in range(12):
        ax = axes[blur_level]
        sigma = sigma_values[blur_level]
        
        if blur_level in all_matrices_data:
            matrix_data = all_matrices_data[blur_level]
            cm = matrix_data['matrix']
            count = matrix_data['count']
            labels = matrix_data['labels']
            
            # Short labels for grid display
            short_labels = ['White', 'Silver', 'Black', 'Blue', 'Other', 'Red', 'Yellow', 'Green']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlBu_r',
                       xticklabels=short_labels, yticklabels=short_labels,
                       ax=ax, cbar=False)
            
            accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
            ax.set_title(f'σ = {sigma:.2f}\nAcc: {accuracy:.3f}', fontweight='bold', fontsize=10)
            
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'σ = {sigma:.2f}\nNo Data', fontweight='bold', fontsize=10)
        
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)
        
        if blur_level >= 8:  # Bottom row
            ax.set_xlabel('Predicted', fontsize=8)
        if blur_level % 4 == 0:  # Left column
            ax.set_ylabel('True', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "confusion_matrices_color_only_grid.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved color-only grid confusion matrices to {output_file}")
    plt.close()

def plot_color_accuracy_trend(all_matrices_data):
    """Plot color classification accuracy trend across blur levels."""
    if not all_matrices_data:
        print("No matrix data to plot trend")
        return
    
    sigma_values = np.linspace(0.0, 4.0, 12)
    accuracies = []
    sigmas = []
    
    for blur_level in range(12):
        if blur_level in all_matrices_data:
            matrix_data = all_matrices_data[blur_level]
            cm = matrix_data['matrix']
            accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
            accuracies.append(accuracy)
            sigmas.append(sigma_values[blur_level])
    
    if not accuracies:
        print("No accuracy data to plot")
        return
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(sigmas, accuracies, 'o-', linewidth=2, markersize=8, color='white', markerfacecolor='blue')
    ax.set_xlabel('Blur Sigma (σ)', fontsize=12)
    ax.set_ylabel('Color Classification Accuracy', fontsize=12)
    ax.set_title('Color Classification Accuracy vs Blur Level', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add value labels on points
    for i, (sigma, acc) in enumerate(zip(sigmas, accuracies)):
        ax.annotate(f'{acc:.3f}', (sigma, acc), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "color_classification_accuracy_trend.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved color classification accuracy trend to {output_file}")
    plt.close()

def generate_color_summary_report(all_matrices_data):
    """Generate a summary report of color classification analysis."""
    print("\n" + "="*80)
    print("COLOR-ONLY CONFUSION MATRIX ANALYSIS SUMMARY")
    print("="*80)
    
    # Calculate sigma values for each blur level
    sigma_values = np.linspace(0.0, 4.0, 12)
    
    for blur_level in range(12):
        sigma = sigma_values[blur_level]
        print(f"\nBlur Level {blur_level} (σ = {sigma:.2f}):")
        print("-" * 40)
        
        if blur_level in all_matrices_data:
            matrix_data = all_matrices_data[blur_level]
            count = matrix_data['count']
            cm = matrix_data['matrix']
            accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
            print(f"  Total samples: {count:4d}, Color accuracy: {accuracy:.3f}")
            
            # Calculate per-color accuracy
            labels = matrix_data['labels']
            print("  Per-color accuracy:")
            for i, label in enumerate(labels):
                if cm[i, i] + sum(cm[i, :]) > 0:
                    color_acc = cm[i, i] / sum(cm[i, :]) if sum(cm[i, :]) > 0 else 0
                    color_count = sum(cm[i, :])
                    print(f"    {label.ljust(12)}: {color_acc:.3f} ({color_count:4d} samples)")
        else:
            print("  No data available")

def main():
    parser = argparse.ArgumentParser(description='Generate color-only confusion matrices')
    parser.add_argument('--blur-level', type=int, choices=range(12), 
                       help='Generate matrices for specific blur level (0-11)')
    parser.add_argument('--all', action='store_true', 
                       help='Generate matrices for all blur levels')
    parser.add_argument('--grid', action='store_true',
                       help='Generate grid visualization')
    parser.add_argument('--trend', action='store_true',
                       help='Generate accuracy trend plot')
    
    args = parser.parse_args()
    
    if not any([args.blur_level is not None, args.all, args.grid, args.trend]):
        parser.error('Must specify --blur-level, --all, --grid, or --trend')
    
    try:
        sigma_values = np.linspace(0.0, 4.0, 12)
        all_matrices_data = {}
        
        if args.grid or args.all or args.trend:
            blur_levels = range(12)
        else:
            blur_levels = [args.blur_level]
        
        for blur_level in blur_levels:
            sigma = sigma_values[blur_level]
            print(f"\nProcessing blur level {blur_level} (σ = {sigma:.2f})...")
            
            # Create confusion matrix for color classification
            matrix_data = create_color_confusion_matrix(blur_level)
            
            if matrix_data:
                all_matrices_data[blur_level] = matrix_data
                if not (args.grid or args.trend):
                    plot_color_confusion_matrix(blur_level, matrix_data, sigma)
            else:
                print(f"No valid predictions found for blur level {blur_level}")
        
        # Generate visualizations if requested
        if args.grid or args.all:
            plot_color_grid(all_matrices_data)
        
        if args.trend or args.all:
            plot_color_accuracy_trend(all_matrices_data)
        
        # Generate summary report
        if args.all:
            generate_color_summary_report(all_matrices_data)
        
        print(f"\nColor-only confusion matrix analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()