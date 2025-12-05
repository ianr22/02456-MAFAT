#!/usr/bin/env python3
"""
All Vehicles Confusion Matrix Analysis for Blur Predictions
Generates confusion matrices for all vehicles combined across all blur levels
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

def create_confusion_matrix_all_vehicles(blur_level):
    """Create confusion matrix for all vehicles combined."""
    # Load predictions
    predictions = load_prediction_data(blur_level)
    if not predictions:
        return None
    
    # Map subclass labels to our reference system
    subclass_mapping = {
        'sub_class_hatchback': 0,      # small vehicle hatchback
        'sub_class_sedan': 1,          # small vehicle sedan  
        'sub_class_pickup': 2,         # small vehicle pickup
        'sub_class_jeep': 3,           # small vehicle jeep
        'sub_class_minivan': 4,        # small vehicle minivan
        'sub_class_light truck': 5,    # large vehicle light truck
        'sub_class_crane truck': 6,    # large vehicle crane truck
        'sub_class_dedicated agricultural vehicle': 7  # large vehicle dedicated agricultural vehicle
    }
    
    # Reference labels (8 subclasses)
    labels = [
        'hatchback',
        'sedan', 
        'pickup',
        'jeep',
        'minivan',
        'light truck',
        'crane truck',
        'agricultural vehicle'
    ]
    
    y_true = []
    y_pred = []
    
    for pred in predictions:
        # Extract true subclass
        true_labels = pred.get('true', [])
        true_subclass = None
        for label in true_labels:
            if label in subclass_mapping:
                true_subclass = subclass_mapping[label]
                break
        
        # Extract predicted subclass (find highest scoring subclass)
        pred_scores = pred.get('pred_scores', {})
        best_subclass = None
        best_score = -1
        
        for subclass, idx in subclass_mapping.items():
            score = pred_scores.get(subclass, 0)
            if score > best_score:
                best_score = score
                best_subclass = idx
        
        if true_subclass is not None and best_subclass is not None:
            y_true.append(true_subclass)
            y_pred.append(best_subclass)
    
    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
        return {
            'matrix': cm,
            'count': len(y_true),
            'labels': labels
        }
    
    return None

def plot_all_vehicles_confusion_matrix(blur_level, matrix_data, sigma_value):
    """Plot confusion matrix for all vehicles."""
    if not matrix_data:
        print(f"No matrix data to plot for blur level {blur_level}")
        return
    
    cm = matrix_data['matrix']
    count = matrix_data['count']
    labels = matrix_data['labels']
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', 
               xticklabels=labels, yticklabels=labels,
               ax=ax, cbar_kws={'shrink': 0.8})
    
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    ax.set_title(f'All Vehicles Confusion Matrix - Blur σ = {sigma_value:.2f}\n({count} samples, accuracy: {accuracy:.3f})', 
                fontweight='bold', fontsize=14)
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    
    # Rotate labels
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', rotation=0, labelsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"confusion_matrix_all_vehicles_blur_{blur_level}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved all vehicles confusion matrix for blur level {blur_level} to {output_file}")
    plt.close()

def plot_all_vehicles_grid(all_matrices_data):
    """Plot grid of all vehicle confusion matrices across blur levels."""
    if not all_matrices_data:
        print("No matrix data to plot")
        return
    
    sigma_values = np.linspace(0.0, 4.0, 12)
    
    # Create grid layout
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    fig.suptitle('All Vehicles Confusion Matrices Across Blur Levels', fontsize=16, fontweight='bold')
    
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
            short_labels = ['Hatch', 'Sedan', 'Pickup', 'Jeep', 'Mini', 'LTruck', 'Crane', 'Agri']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr',
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
    output_file = "confusion_matrices_all_vehicles_grid.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved all vehicles grid confusion matrices to {output_file}")
    plt.close()

def generate_all_vehicles_summary_report(all_matrices_data):
    """Generate a summary report of all vehicles analysis."""
    print("\n" + "="*80)
    print("ALL VEHICLES CONFUSION MATRIX ANALYSIS SUMMARY")
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
            print(f"  Total samples: {count:4d}, Overall accuracy: {accuracy:.3f}")
        else:
            print("  No data available")

def main():
    parser = argparse.ArgumentParser(description='Generate all vehicles confusion matrices')
    parser.add_argument('--blur-level', type=int, choices=range(12), 
                       help='Generate matrices for specific blur level (0-11)')
    parser.add_argument('--all', action='store_true', 
                       help='Generate matrices for all blur levels')
    parser.add_argument('--grid', action='store_true',
                       help='Generate grid visualization')
    
    args = parser.parse_args()
    
    if not any([args.blur_level is not None, args.all, args.grid]):
        parser.error('Must specify --blur-level, --all, or --grid')
    
    try:
        sigma_values = np.linspace(0.0, 4.0, 12)
        all_matrices_data = {}
        
        if args.grid or args.all:
            blur_levels = range(12)
        else:
            blur_levels = [args.blur_level]
        
        for blur_level in blur_levels:
            sigma = sigma_values[blur_level]
            print(f"\nProcessing blur level {blur_level} (σ = {sigma:.2f})...")
            
            # Create confusion matrix for all vehicles
            matrix_data = create_confusion_matrix_all_vehicles(blur_level)
            
            if matrix_data:
                all_matrices_data[blur_level] = matrix_data
                if not args.grid:
                    plot_all_vehicles_confusion_matrix(blur_level, matrix_data, sigma)
            else:
                print(f"No valid predictions found for blur level {blur_level}")
        
        # Generate grid visualization if requested
        if args.grid or args.all:
            plot_all_vehicles_grid(all_matrices_data)
        
        # Generate summary report
        if args.all:
            generate_all_vehicles_summary_report(all_matrices_data)
        
        print(f"\nAll vehicles confusion matrix analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()