#!/usr/bin/env python3
"""
Generate confusion matrices for all blur levels (0-11) with enhanced color visualization.
Usage: python results/all_blur_confusion_matrices.py --results-dir results --out-dir results/plots
"""

import os
import glob
import json
import argparse
from collections import Counter
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_json_file(path):
    """Load JSON data from file, handling both single JSON and JSONL formats"""
    with open(path, 'r') as f:
        try:
            data = json.load(f)
        except Exception:
            data = []
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
    return data


def find_json_for_blur(results_dir, level):
    """Find predictions JSON file for a specific blur level"""
    pattern = os.path.join(results_dir, f"blur_{level}_run_*", "predictions_with_scores.json")
    matches = glob.glob(pattern)
    if not matches:
        # Try alternate pattern
        alternate = os.path.join(results_dir, f"blur_{level}", "predictions_with_scores.json")
        matches = glob.glob(alternate)
    return matches[0] if matches else None


def extract_predictions_and_labels(data):
    """Extract predicted and true labels from the data"""
    trues, preds = [], []
    
    for rec in data:
        if not isinstance(rec, dict):
            continue
            
        tr = rec.get('true') or rec.get('ground_truth') or rec.get('gt')
        pr = rec.get('pred') or rec.get('predictions') or rec.get('prediction')
        scores = rec.get('pred_scores') or rec.get('scores') or rec.get('probs')

        # Get top predicted label
        top_label = None
        if isinstance(pr, list) and pr:
            top_label = pr[0]
        elif isinstance(scores, dict) and scores:
            try:
                top_label = max(scores, key=lambda k: float(scores[k]))
            except Exception:
                top_label = None

        # Get first true label
        true_first = None
        if isinstance(tr, list) and tr:
            true_first = tr[0]
        elif isinstance(tr, str):
            true_first = tr

        if top_label is not None and true_first is not None:
            preds.append(top_label)
            trues.append(true_first)
    
    return trues, preds


def get_reference_labels():
    """Get the standard reference labels for consistent ordering"""
    return [
        'general_class_small vehicle',
        'luggage_carrier', 
        'sunroof',
        'open_cargo_area',
        'wrecked',
        'spare_wheel',
        'enclosed_cab',
        'general_class_large vehicle'
    ]


def create_single_confusion_matrix(results_dir, blur_level, labels, out_dir):
    """Create confusion matrix for a single blur level"""
    path = find_json_for_blur(results_dir, blur_level)
    if not path:
        print(f"No data found for blur {blur_level}")
        return None
        
    data = load_json_file(path)
    trues, preds = extract_predictions_and_labels(data)
    
    if not trues:
        print(f"No valid predictions found for blur {blur_level}")
        return None
    
    # Create confusion matrix
    cm = confusion_matrix(trues, preds, labels=labels, normalize='true')
    
    # Map blur level to sigma value
    num_levels = 12
    sigmas = np.linspace(0.0, 4.0, num_levels)
    sigma_value = sigmas[blur_level] if blur_level < len(sigmas) else blur_level
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Format labels for display
    display_labels = [label.replace('_', ' ') for label in labels]
    
    # Create heatmap with enhanced colors
    sns.heatmap(
        cm,
        xticklabels=display_labels,
        yticklabels=display_labels,
        cmap='RdYlBu_r',  # Red-Yellow-Blue reversed for better contrast
        annot=True,
        fmt='.2f',
        square=True,
        vmin=0,
        vmax=1,
        linewidths=1,
        linecolor='white',
        cbar_kws={'label': 'Normalized True Positive Rate', 'shrink': 0.8}
    )
    
    plt.title(f'Confusion Matrix - Blur Level {blur_level} (σ = {sigma_value:.2f})', 
              fontsize=14, pad=20)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Save individual matrix
    output_path = os.path.join(out_dir, f'confusion_matrix_blur_{blur_level}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"Saved confusion matrix for blur {blur_level} -> {output_path}")
    return cm


def create_all_blur_grid(results_dir, out_dir, levels=None):
    """Create a grid of confusion matrices for all blur levels"""
    if levels is None:
        levels = list(range(0, 12))  # All 12 blur levels
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Get reference labels
    labels = get_reference_labels()
    
    # Determine grid size
    n_levels = len(levels)
    ncols = 4  # 4 columns for better layout with 12 levels
    nrows = math.ceil(n_levels / ncols)
    
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows)
    )
    
    # Ensure axes is always 2D
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Map blur levels to sigma values
    num_levels = 12
    sigmas = np.linspace(0.0, 4.0, num_levels)
    
    cms = []
    
    for idx, blur_level in enumerate(levels):
        ax = axes[idx]
        
        # Get data for this blur level
        path = find_json_for_blur(results_dir, blur_level)
        if not path:
            ax.axis('off')
            ax.text(0.5, 0.5, f'No data\nBlur {blur_level}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
            
        data = load_json_file(path)
        trues, preds = extract_predictions_and_labels(data)
        
        if not trues:
            ax.axis('off')
            ax.text(0.5, 0.5, f'No predictions\nBlur {blur_level}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create confusion matrix
        cm = confusion_matrix(trues, preds, labels=labels, normalize='true')
        cms.append(cm)
        
        # Format labels for display
        display_labels = [label.replace('_', ' ') for label in labels]
        
        # Get sigma value
        sigma_value = sigmas[blur_level] if blur_level < len(sigmas) else blur_level
        
        # Create heatmap
        sns.heatmap(
            cm,
            ax=ax,
            xticklabels=display_labels,
            yticklabels=display_labels,
            cmap='RdYlBu_r',
            cbar=False,  # Individual colorbars would be too cluttered
            square=True,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor='white'
        )
        
        ax.set_title(f'σ = {sigma_value:.2f}', fontsize=12, pad=10)
        ax.set_xlabel('Predicted' if idx >= (nrows-1)*ncols else '', fontsize=10)
        ax.set_ylabel('True' if idx % ncols == 0 else '', fontsize=10)
        
        # Rotate labels for better readability
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        ax.tick_params(axis='y', labelsize=8, rotation=0)
    
    # Turn off unused subplots
    for ax in axes[n_levels:]:
        ax.axis('off')
    
    # Add a single colorbar for the entire figure
    if cms:
        # Create colorbar axes
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(0, 1)), 
            cax=cbar_ax
        )
        cbar.set_label('Normalized True Positive Rate', rotation=270, labelpad=20)
    
    # Adjust layout
    plt.suptitle('Confusion Matrices for All Blur Levels', fontsize=16, y=0.95)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.90, 
                       wspace=0.3, hspace=0.4)
    
    # Save grid
    grid_path = os.path.join(out_dir, 'confusion_matrices_all_blur_levels_grid.png')
    fig.savefig(grid_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    
    print(f"Saved grid of all confusion matrices -> {grid_path}")


def create_selected_blur_comparison(results_dir, out_dir, selected_levels=[0, 3, 8, 11]):
    """Create comparison of selected blur levels with enhanced styling"""
    os.makedirs(out_dir, exist_ok=True)
    
    labels = get_reference_labels()
    
    # Create 2x2 grid for selected levels
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Map blur levels to sigma values
    num_levels = 12
    sigmas = np.linspace(0.0, 4.0, num_levels)
    
    for idx, blur_level in enumerate(selected_levels):
        ax = axes[idx]
        
        path = find_json_for_blur(results_dir, blur_level)
        if not path:
            ax.axis('off')
            continue
            
        data = load_json_file(path)
        trues, preds = extract_predictions_and_labels(data)
        
        if not trues:
            ax.axis('off')
            continue
        
        cm = confusion_matrix(trues, preds, labels=labels, normalize='true')
        display_labels = [label.replace('_', ' ') for label in labels]
        sigma_value = sigmas[blur_level] if blur_level < len(sigmas) else blur_level
        
        # Enhanced heatmap with different color scheme
        heatmap = sns.heatmap(
            cm,
            ax=ax,
            xticklabels=display_labels,
            yticklabels=display_labels,
            cmap='plasma',  # Different colormap for variety
            cbar=False,
            square=True,
            vmin=0,
            vmax=1,
            linewidths=1,
            linecolor='white'
        )
        
        # Store first heatmap for colorbar
        if idx == 0:
            first_mappable = heatmap.collections[0]
        
        ax.set_title(f'σ = {sigma_value:.2f}', fontsize=14, pad=15)
        
        # Conditional axis labels (similar to your previous setup)
        if idx == 0:  # Top-left: only y-axis
            ax.set_xlabel('')
            ax.set_ylabel('True', fontsize=11)
            ax.set_xticklabels([])
            ax.tick_params(bottom=False)
        elif idx == 1:  # Top-right: no axes
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)
        elif idx == 2:  # Bottom-left: both axes
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('True', fontsize=11)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
            plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, ha='right', fontsize=9)
        elif idx == 3:  # Bottom-right: only x-axis
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('')
            ax.set_yticklabels([])
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
            ax.tick_params(left=False)
    
    # Add colorbar
    if 'first_mappable' in locals():
        cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
        cbar = fig.colorbar(first_mappable, cax=cbar_ax)
        cbar.set_label('Normalized True Positive Rate', rotation=270, labelpad=20, fontsize=11)
    
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.10, top=0.95, wspace=0.8, hspace=0.4)
    
    # Save comparison
    comparison_path = os.path.join(out_dir, 'confusion_matrices_selected_blur_comparison.png')
    fig.savefig(comparison_path, dpi=300, bbox_inches='tight', pad_inches=0.4)
    plt.close()
    
    print(f"Saved selected blur comparison -> {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrices for all blur levels')
    parser.add_argument('--results-dir', default='results', help='Base results directory')
    parser.add_argument('--out-dir', default='results/plots', help='Output directory for plots')
    parser.add_argument('--levels', default=None, help='Comma-separated blur levels (default: all 0-11)')
    parser.add_argument('--individual', action='store_true', help='Generate individual matrices')
    parser.add_argument('--grid', action='store_true', help='Generate grid of all matrices')
    parser.add_argument('--comparison', action='store_true', help='Generate selected blur comparison')
    parser.add_argument('--all', action='store_true', help='Generate all visualizations')
    
    args = parser.parse_args()
    
    # Parse levels
    if args.levels:
        levels = [int(x) for x in args.levels.split(',')]
    else:
        levels = list(range(0, 12))  # All blur levels
    
    # Determine what to generate
    generate_individual = args.individual or args.all
    generate_grid = args.grid or args.all
    generate_comparison = args.comparison or args.all
    
    # If no specific option is given, generate all
    if not (args.individual or args.grid or args.comparison):
        generate_individual = generate_grid = generate_comparison = True
    
    print(f"Processing blur levels: {levels}")
    
    # Get reference labels
    labels = get_reference_labels()
    
    # Generate individual matrices
    if generate_individual:
        print("\n=== Generating Individual Confusion Matrices ===")
        for level in levels:
            create_single_confusion_matrix(args.results_dir, level, labels, args.out_dir)
    
    # Generate grid of all matrices
    if generate_grid:
        print("\n=== Generating Grid of All Confusion Matrices ===")
        create_all_blur_grid(args.results_dir, args.out_dir, levels)
    
    # Generate selected blur comparison
    if generate_comparison:
        print("\n=== Generating Selected Blur Comparison ===")
        selected_levels = [0, 3, 8, 11] if len(levels) >= 4 else levels[:4]
        create_selected_blur_comparison(args.results_dir, args.out_dir, selected_levels)
    
    print(f"\n✅ All confusion matrices saved to: {args.out_dir}")


if __name__ == '__main__':
    main()