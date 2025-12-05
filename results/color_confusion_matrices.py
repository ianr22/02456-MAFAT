#!/usr/bin/env python3
"""
Color-Based Confusion Matrix Analysis for Blur Predictions
Generates confusion matrices segmented by vehicle color subcategories
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from pathlib import Path
import argparse

def load_csv_data():
    """Load the training CSV file to get color information."""
    csv_path = Path("../dataset_v2/train.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # Create mapping from tag_id to color
    color_mapping = dict(zip(df['tag_id'], df['color']))
    return color_mapping, df

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

def extract_image_id_from_filename(filename):
    """Extract image_id from filename like '16490_15036.png'."""
    # Remove .png extension and split by underscore
    base = filename.replace('.png', '')
    parts = base.split('_')
    if len(parts) >= 2:
        return int(parts[1])  # Return the image_id part
    return None

def get_predictions_by_color(predictions, color_mapping):
    """Group predictions by color categories."""
    color_predictions = {}
    
    # Create reverse mapping from image_id to tag_id and color
    # This requires checking which tag_id corresponds to each image_id
    # For now, let's use the predicted and true color directly from the JSON
    
    for pred in predictions:
        # Extract true color from the prediction data
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

def create_confusion_matrix_by_color(blur_level, color_mapping):
    """Create confusion matrices for each color category."""
    # Load predictions
    predictions = load_prediction_data(blur_level)
    if not predictions:
        return None
    
    # Group by color
    color_predictions = get_predictions_by_color(predictions, color_mapping)
    
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
    
    color_matrices = {}
    
    for color, color_preds in color_predictions.items():
        if len(color_preds) < 5:  # Skip colors with too few samples
            continue
            
        y_true = []
        y_pred = []
        
        for pred in color_preds:
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
            color_matrices[color] = {
                'matrix': cm,
                'count': len(y_true),
                'labels': labels
            }
    
    return color_matrices

def plot_color_confusion_matrices(blur_level, color_matrices, sigma_value):
    """Plot confusion matrices for each color category."""
    if not color_matrices:
        print(f"No color matrices to plot for blur level {blur_level}")
        return
    
    # Filter colors by minimum sample count and sort by count
    min_samples = 20
    filtered_colors = {
        color: data for color, data in color_matrices.items() 
        if data['count'] >= min_samples
    }
    
    if not filtered_colors:
        print(f"No colors with sufficient samples (>= {min_samples}) for blur level {blur_level}")
        return
    
    # Sort colors by sample count (descending)
    sorted_colors = sorted(filtered_colors.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # Calculate grid size
    n_colors = len(sorted_colors)
    cols = min(3, n_colors)
    rows = (n_colors + cols - 1) // cols
    
    # Create the plot
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    fig.suptitle(f'Confusion Matrices by Color - Blur σ = {sigma_value:.2f}', fontsize=16, fontweight='bold')
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Shortened labels for display
    short_labels = [
        'Hatchback',
        'Sedan', 
        'Pickup',
        'Jeep',
        'Minivan',
        'Light Truck',
        'Crane Truck',
        'Agricultural'
    ]
    
    for idx, (color, data) in enumerate(sorted_colors):
        ax = axes[idx]
        cm = data['matrix']
        count = data['count']
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=short_labels, yticklabels=short_labels,
                   ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(f'{color.title()}\n({count} samples)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Predicted Class', fontsize=10)
        ax.set_ylabel('True Class', fontsize=10)
        
        # Rotate labels
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)
    
    # Hide unused subplots
    for idx in range(len(sorted_colors), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"confusion_matrices_color_blur_{blur_level}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved color confusion matrices for blur level {blur_level} to {output_file}")
    plt.close()

def generate_color_summary_report(all_color_data):
    """Generate a summary report of color-based analysis."""
    print("\n" + "="*80)
    print("COLOR-BASED CONFUSION MATRIX ANALYSIS SUMMARY")
    print("="*80)
    
    # Calculate sigma values for each blur level
    sigma_values = np.linspace(0.0, 4.0, 12)
    
    for blur_level in range(12):
        sigma = sigma_values[blur_level]
        print(f"\nBlur Level {blur_level} (σ = {sigma:.2f}):")
        print("-" * 40)
        
        if blur_level in all_color_data and all_color_data[blur_level]:
            color_data = all_color_data[blur_level]
            
            # Sort colors by sample count
            sorted_colors = sorted(color_data.items(), key=lambda x: x[1]['count'], reverse=True)
            
            for color, data in sorted_colors:
                count = data['count']
                cm = data['matrix']
                accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
                print(f"  {color.ljust(15)}: {count:4d} samples, accuracy: {accuracy:.3f}")
        else:
            print("  No data available")

def main():
    parser = argparse.ArgumentParser(description='Generate color-based confusion matrices')
    parser.add_argument('--blur-level', type=int, choices=range(12), 
                       help='Generate matrices for specific blur level (0-11)')
    parser.add_argument('--all', action='store_true', 
                       help='Generate matrices for all blur levels')
    parser.add_argument('--summary-only', action='store_true',
                       help='Generate summary report only')
    
    args = parser.parse_args()
    
    if not any([args.blur_level is not None, args.all, args.summary_only]):
        parser.error('Must specify --blur-level, --all, or --summary-only')
    
    try:
        # Load color mapping from CSV
        print("Loading color mapping from CSV...")
        color_mapping, csv_df = load_csv_data()
        print(f"Loaded color data for {len(color_mapping)} vehicles")
        
        # Show color distribution
        color_counts = csv_df['color'].value_counts()
        print("\nColor distribution in dataset:")
        for color, count in color_counts.items():
            print(f"  {color.ljust(15)}: {count:4d} vehicles")
        
        sigma_values = np.linspace(0.0, 4.0, 12)
        all_color_data = {}
        
        if args.summary_only:
            # Load data for all blur levels for summary
            for blur_level in range(12):
                print(f"\nProcessing blur level {blur_level}...")
                color_matrices = create_confusion_matrix_by_color(blur_level, color_mapping)
                if color_matrices:
                    all_color_data[blur_level] = color_matrices
            
            generate_color_summary_report(all_color_data)
            return
        
        if args.all:
            blur_levels = range(12)
        else:
            blur_levels = [args.blur_level]
        
        for blur_level in blur_levels:
            sigma = sigma_values[blur_level]
            print(f"\nProcessing blur level {blur_level} (σ = {sigma:.2f})...")
            
            # Create confusion matrices by color
            color_matrices = create_confusion_matrix_by_color(blur_level, color_mapping)
            
            if color_matrices:
                all_color_data[blur_level] = color_matrices
                plot_color_confusion_matrices(blur_level, color_matrices, sigma)
            else:
                print(f"No valid predictions found for blur level {blur_level}")
        
        # Generate summary report
        if args.all:
            generate_color_summary_report(all_color_data)
        
        print(f"\nColor-based confusion matrix analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()