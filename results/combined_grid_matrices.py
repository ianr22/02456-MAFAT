#!/usr/bin/env python3
"""
Combined Grid Confusion Matrix Analysis
Combines blur levels 0, 3, 8, and 11 into single grid visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from pathlib import Path
import argparse

def get_available_blur_levels():
    """Dynamically detect available blur levels from training_prediction folder."""
    blur_dirs = list(Path("training_prediction").glob("blur_*_run_*"))
    blur_levels = []
    
    for blur_dir in blur_dirs:
        # Extract blur level from directory name
        dir_name = blur_dir.name
        if dir_name.startswith('blur_') and '_run_' in dir_name:
            blur_level_str = dir_name.split('_')[1]
            try:
                blur_level = int(blur_level_str)
                blur_levels.append(blur_level)
            except ValueError:
                continue
    
    return sorted(list(set(blur_levels)))

def calculate_sigma_range():
    """Calculate sigma range based on available blur levels."""
    available_levels = get_available_blur_levels()
    if not available_levels:
        return np.linspace(0.0, 4.0, 12)  # Fallback
    
    max_blur = max(available_levels)
    # Assume sigma ranges from 0 to 8 based on the max blur level
    return np.linspace(0.0, 8.0, max_blur + 1)

def select_representative_blur_levels(available_levels, num_levels=4):
    """Select representative blur levels for 2x2 grid."""
    if len(available_levels) < num_levels:
        return available_levels
    
    # Select evenly spaced levels including min and max
    indices = np.linspace(0, len(available_levels) - 1, num_levels, dtype=int)
    return [available_levels[i] for i in indices]

def load_prediction_data(blur_level):
    """Load prediction data for a specific blur level."""
    blur_dirs = list(Path("training_prediction").glob(f"blur_{blur_level}_run_*"))
    
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
    
    return all_predictions

def create_vehicle_confusion_matrix(blur_level):
    """Create confusion matrix for vehicle classification."""
    predictions = load_prediction_data(blur_level)
    if not predictions:
        return None
    
    subclass_mapping = {
        'sub_class_hatchback': 0,
        'sub_class_sedan': 1,
        'sub_class_pickup': 2,
        'sub_class_jeep': 3,
        'sub_class_minivan': 4,
        'sub_class_light truck': 5,
        'sub_class_crane truck': 6,
        'sub_class_dedicated agricultural vehicle': 7
    }
    
    labels = ['Hatch', 'Sedan', 'Pickup', 'Jeep', 'Mini', 'LTruck', 'Crane', 'Agri']
    
    y_true = []
    y_pred = []
    
    for pred in predictions:
        true_labels = pred.get('true', [])
        true_subclass = None
        for label in true_labels:
            if label in subclass_mapping:
                true_subclass = subclass_mapping[label]
                break
        
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
        accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
        return {'matrix': cm, 'count': len(y_true), 'labels': labels, 'accuracy': accuracy}
    
    return None

def create_color_confusion_matrix(blur_level):
    """Create confusion matrix for color classification."""
    predictions = load_prediction_data(blur_level)
    if not predictions:
        return None
    
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
    
    labels = ['White', 'Silver', 'Black', 'Blue', 'Other', 'Red', 'Yellow', 'Green']
    
    y_true = []
    y_pred = []
    
    for pred in predictions:
        true_labels = pred.get('true', [])
        true_color = None
        for label in true_labels:
            if label in color_mapping:
                true_color = color_mapping[label]
                break
        
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
        accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
        return {'matrix': cm, 'count': len(y_true), 'labels': labels, 'accuracy': accuracy}
    
    return None

def get_predictions_by_color(predictions):
    """Group predictions by color categories."""
    color_predictions = {}
    
    for pred in predictions:
        true_labels = pred.get('true', [])
        true_color = None
        
        for label in true_labels:
            if label.startswith('color_'):
                true_color = label.replace('color_', '')
                break
        
        if true_color:
            if true_color not in color_predictions:
                color_predictions[true_color] = []
            color_predictions[true_color].append(pred)
    
    return color_predictions

def create_color_based_confusion_matrices(blur_level):
    """Create confusion matrices segmented by color."""
    predictions = load_prediction_data(blur_level)
    if not predictions:
        return None
    
    color_predictions = get_predictions_by_color(predictions)
    
    subclass_mapping = {
        'sub_class_hatchback': 0,
        'sub_class_sedan': 1,
        'sub_class_pickup': 2,
        'sub_class_jeep': 3,
        'sub_class_minivan': 4,
        'sub_class_light truck': 5,
        'sub_class_crane truck': 6,
        'sub_class_dedicated agricultural vehicle': 7
    }
    
    labels = ['Hatch', 'Sedan', 'Pickup', 'Jeep', 'Mini', 'LTruck', 'Crane', 'Agri']
    
    color_matrices = {}
    
    for color, color_preds in color_predictions.items():
        if len(color_preds) < 20:  # Skip colors with too few samples
            continue
            
        y_true = []
        y_pred = []
        
        for pred in color_preds:
            true_labels = pred.get('true', [])
            true_subclass = None
            for label in true_labels:
                if label in subclass_mapping:
                    true_subclass = subclass_mapping[label]
                    break
            
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
            accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
            color_matrices[color] = {
                'matrix': cm,
                'count': len(y_true),
                'labels': labels,
                'accuracy': accuracy
            }
    
    return color_matrices

def plot_combined_vehicle_grid():
    """Plot combined grid for vehicle classification in 2x2 format."""
    available_levels = get_available_blur_levels()
    blur_levels = select_representative_blur_levels(available_levels)
    sigma_values = calculate_sigma_range()
    
    print(f"Using blur levels: {blur_levels}")
    print(f"Available blur levels: {available_levels}")
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Vehicle Type Classification - Selected Blur Levels', fontsize=16, fontweight='bold')
    
    # Position mapping for 2x2 grid
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # Add shared colorbar with more spacing
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])

    for i, blur_level in enumerate(blur_levels):
        row, col = positions[i]
        ax = axes[row, col]
        sigma = sigma_values[blur_level]
        
        matrix_data = create_vehicle_confusion_matrix(blur_level)
        
        if matrix_data:
            cm = matrix_data['matrix']
            labels = matrix_data['labels']
            
            # Create heatmap with colorbar only on the last plot
            im = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=labels, yticklabels=labels,
                           ax=ax, cbar=(i == 3), cbar_ax=cbar_ax if i == 3 else None,
                           vmin=0, vmax=np.max([create_vehicle_confusion_matrix(bl)['matrix'].max() 
                                               for bl in blur_levels if create_vehicle_confusion_matrix(bl)]))
            
            ax.set_title(f'σ = {sigma:.2f}', fontweight='bold', fontsize=14)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'σ = {sigma:.2f}', fontweight='bold', fontsize=14)
        
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=10)
        
        # Mute specific axes based on blur level
        if blur_level == 0:  # Top left - mute x axis
            ax.set_xticklabels([])
        elif blur_level == 3:  # Top right - mute both x and y axes
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        elif blur_level == 11:  # Bottom right - mute y axis
            ax.set_yticklabels([])
        
        # Add labels only on bottom and left edges (but respect muting)
        if row == 1 and blur_level not in [0, 3]:  # Bottom row, not muted
            ax.set_xlabel('Predicted', fontsize=12)
        if col == 0 and blur_level not in [3, 11]:  # Left column, not muted
            ax.set_ylabel('True', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig('confusion_matrices_vehicle_combined_grid.png', dpi=300, bbox_inches='tight')
    print("Saved combined vehicle confusion matrices to confusion_matrices_vehicle_combined_grid.png")
    plt.close()

def plot_combined_color_grid():
    """Plot combined grid for color classification in 2x2 format."""
    available_levels = get_available_blur_levels()
    blur_levels = select_representative_blur_levels(available_levels)
    sigma_values = calculate_sigma_range()
    
    print(f"Using blur levels: {blur_levels}")
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Color Classification - Selected Blur Levels', fontsize=16, fontweight='bold')
    
    # Position mapping for 2x2 grid
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # Add shared colorbar with more spacing
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])

    # Collect all matrices to determine global vmin/vmax
    all_matrices = []
    for blur_level in blur_levels:
        matrix_data = create_color_confusion_matrix(blur_level)
        if matrix_data:
            all_matrices.append(matrix_data['matrix'])
    
    if all_matrices:
        vmax = max([cm.max() for cm in all_matrices])
    else:
        vmax = 1
    
    for i, blur_level in enumerate(blur_levels):
        row, col = positions[i]
        ax = axes[row, col]
        sigma = sigma_values[blur_level]
        
        matrix_data = create_color_confusion_matrix(blur_level)
        
        if matrix_data:
            cm = matrix_data['matrix']
            labels = matrix_data['labels']
            
            # Create heatmap with colorbar only on the last plot
            im = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=labels, yticklabels=labels,
                           ax=ax, cbar=(i == 3), cbar_ax=cbar_ax if i == 3 else None,
                           vmin=0, vmax=vmax)
            
            ax.set_title(f'σ = {sigma:.2f}', fontweight='bold', fontsize=14)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'σ = {sigma:.2f}', fontweight='bold', fontsize=14)
        
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=10)
        
        # Mute specific axes based on blur level
        if blur_level == 0:  # Top left - mute x axis
            ax.set_xticklabels([])
        elif blur_level == 3:  # Top right - mute both x and y axes
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        elif blur_level == 11:  # Bottom right - mute y axis
            ax.set_yticklabels([])
        
        # Add labels only on bottom and left edges (but respect muting)
        if row == 1 and blur_level not in [0, 3]:  # Bottom row, not muted
            ax.set_xlabel('Predicted', fontsize=12)
        if col == 0 and blur_level not in [3, 11]:  # Left column, not muted
            ax.set_ylabel('True', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig('confusion_matrices_color_combined_grid.png', dpi=300, bbox_inches='tight')
    print("Saved combined color confusion matrices to confusion_matrices_color_combined_grid.png")
    plt.close()

def plot_combined_color_based_grid():
    """Plot combined grid for color-based vehicle classification."""
    available_levels = get_available_blur_levels()
    blur_levels = select_representative_blur_levels(available_levels)
    sigma_values = calculate_sigma_range()
    
    print(f"Using blur levels: {blur_levels}")
    
    # Get top colors across all blur levels
    all_colors = set()
    for blur_level in blur_levels:
        color_matrices = create_color_based_confusion_matrices(blur_level)
        if color_matrices:
            all_colors.update(color_matrices.keys())
    
    # Sort colors by total samples (use blur level 0 as reference)
    color_order = []
    reference_matrices = create_color_based_confusion_matrices(0)
    if reference_matrices:
        sorted_colors = sorted(reference_matrices.items(), 
                             key=lambda x: x[1]['count'], reverse=True)
        color_order = [color for color, _ in sorted_colors[:6]]  # Top 6 colors
    
    if not color_order:
        print("No color data available for color-based grid")
        return
    
    fig, axes = plt.subplots(len(color_order), 4, figsize=(16, 4*len(color_order)))
    fig.suptitle('Vehicle Classification by Color - Selected Blur Levels', 
                fontsize=16, fontweight='bold')
    
    if len(color_order) == 1:
        axes = axes.reshape(1, -1)
    
    for row, color in enumerate(color_order):
        for col, blur_level in enumerate(blur_levels):
            ax = axes[row, col]
            sigma = sigma_values[blur_level]
            
            color_matrices = create_color_based_confusion_matrices(blur_level)
            
            if color_matrices and color in color_matrices:
                matrix_data = color_matrices[color]
                cm = matrix_data['matrix']
                labels = matrix_data['labels']
                accuracy = matrix_data['accuracy']
                count = matrix_data['count']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=labels, yticklabels=labels,
                           ax=ax, cbar=False)
                
                if row == 0:  # Top row
                    ax.set_title(f'σ = {sigma:.2f}', fontweight='bold', fontsize=11)
                
                if col == 3:  # Right column
                    ax.text(1.05, 0.5, f'{color.title()}\nAcc: {accuracy:.3f}\n({count} samples)', 
                           transform=ax.transAxes, rotation=90, va='center', 
                           fontweight='bold', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'Insufficient\nData', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
                if row == 0:
                    ax.set_title(f'σ = {sigma:.2f}', fontweight='bold', fontsize=11)
                if col == 3:
                    ax.text(1.05, 0.5, f'{color.title()}\nNo Data', 
                           transform=ax.transAxes, rotation=90, va='center', 
                           fontweight='bold', fontsize=10)
            
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', rotation=0, labelsize=8)
            
            if row == len(color_order) - 1:  # Bottom row
                ax.set_xlabel('Predicted', fontsize=9)
            if col == 0:  # Left column
                ax.set_ylabel('True', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_color_based_combined_grid.png', dpi=300, bbox_inches='tight')
    print("Saved combined color-based confusion matrices to confusion_matrices_color_based_combined_grid.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate combined grid confusion matrices for blur levels 0, 3, 8, 11')
    parser.add_argument('--type', choices=['vehicle', 'color', 'color-based', 'all'], 
                       default='all', help='Type of confusion matrix to generate')
    
    args = parser.parse_args()
    
    try:
        available_levels = get_available_blur_levels()
        selected_levels = select_representative_blur_levels(available_levels)
        print(f"Generating combined confusion matrix grids for blur levels {selected_levels}...")
        print(f"Available blur levels: {available_levels}")
        
        if args.type in ['vehicle', 'all']:
            print("\nGenerating vehicle classification grid...")
            plot_combined_vehicle_grid()
        
        if args.type in ['color', 'all']:
            print("\nGenerating color classification grid...")
            plot_combined_color_grid()
        
        if args.type in ['color-based', 'all']:
            print("\nGenerating color-based vehicle classification grid...")
            plot_combined_color_based_grid()
        
        print("\nCombined grid generation complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()