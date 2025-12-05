#!/usr/bin/env python3
"""
Accuracy Trend Analysis
Extract accuracy metrics and create trend plots for color and vehicle classification
across the full sigma range 0-8
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

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

def load_accuracy_data():
    """Load accuracy data from all blur level directories."""
    accuracy_data = []
    available_levels = get_available_blur_levels()
    max_blur_level = max(available_levels) if available_levels else 23
    
    print(f"Found blur levels: {available_levels}")
    
    # Check all available blur levels
    for blur_level in available_levels:
        blur_dirs = list(Path("training_prediction").glob(f"blur_{blur_level}_run_*"))
        
        if blur_dirs:
            blur_dir = blur_dirs[0]  # Use first matching directory
            metrics_file = blur_dir / "metrics.json"
            
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    # Calculate sigma value dynamically
                    sigma = blur_level * 8.0 / max_blur_level  # Map to 0-8 range
                    
                    accuracy_data.append({
                        'blur_level': blur_level,
                        'sigma': sigma,
                        'metrics': metrics
                    })
                    print(f"Loaded metrics for blur level {blur_level}, sigma = {sigma:.3f}")
                    
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading {metrics_file}: {e}")
    
    return accuracy_data

def calculate_classification_accuracies(blur_level):
    """Calculate color and vehicle classification accuracies from prediction data."""
    blur_dirs = list(Path("training_prediction").glob(f"blur_{blur_level}_run_*"))
    
    if not blur_dirs:
        return None, None
    
    predictions_file = blur_dirs[0] / "predictions_with_scores.json"
    
    if not predictions_file.exists():
        return None, None
    
    try:
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None, None
    
    color_correct = 0
    vehicle_correct = 0
    total_samples = len(predictions)
    
    for pred in predictions:
        predicted = pred.get('pred', [])
        true_labels = pred.get('true', [])
        
        # Check color accuracy
        pred_color = None
        true_color = None
        
        for p in predicted:
            if p.startswith('color_'):
                pred_color = p
                break
        
        for t in true_labels:
            if t.startswith('color_'):
                true_color = t
                break
        
        if pred_color and true_color and pred_color == true_color:
            color_correct += 1
        
        # Check vehicle subclass accuracy
        pred_vehicle = None
        true_vehicle = None
        
        for p in predicted:
            if p.startswith('sub_class_'):
                pred_vehicle = p
                break
        
        for t in true_labels:
            if t.startswith('sub_class_'):
                true_vehicle = t
                break
        
        if pred_vehicle and true_vehicle and pred_vehicle == true_vehicle:
            vehicle_correct += 1
    
    if total_samples > 0:
        color_accuracy = color_correct / total_samples
        vehicle_accuracy = vehicle_correct / total_samples
        return color_accuracy, vehicle_accuracy
    
    return None, None

def extract_classification_accuracies(accuracy_data):
    """Extract color and vehicle classification accuracies."""
    color_accuracies = []
    vehicle_accuracies = []
    sigma_values = []
    
    for data in accuracy_data:
        blur_level = data['blur_level']
        sigma = data['sigma']
        
        # Calculate accuracies from prediction data
        color_acc, vehicle_acc = calculate_classification_accuracies(blur_level)
        
        if color_acc is not None and vehicle_acc is not None:
            sigma_values.append(sigma)
            color_accuracies.append(color_acc)
            vehicle_accuracies.append(vehicle_acc)
            print(f"Sigma {sigma:.3f}: Color={color_acc:.3f}, Vehicle={vehicle_acc:.3f}")
    
    return np.array(sigma_values), np.array(color_accuracies), np.array(vehicle_accuracies)

def plot_accuracy_trends(sigma_values, color_accuracies, vehicle_accuracies):
    """Create accuracy trend plots for both classifications."""
    
    # Sort by sigma values
    sort_indices = np.argsort(sigma_values)
    sigma_sorted = sigma_values[sort_indices]
    color_sorted = color_accuracies[sort_indices]
    vehicle_sorted = vehicle_accuracies[sort_indices]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color Classification Accuracy Plot
    ax1.plot(sigma_sorted, color_sorted, 'bo-', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Blur Sigma (σ)', fontsize=12)
    ax1.set_ylabel('Color Classification Accuracy', fontsize=12)
    ax1.set_title('Color Classification Accuracy vs Blur Level', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.2, 8.2)
    ax1.set_ylim(0, 1.0)
    
    # Vehicle Classification Accuracy Plot
    ax2.plot(sigma_sorted, vehicle_sorted, 'ro-', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Blur Sigma (σ)', fontsize=12)
    ax2.set_ylabel('Vehicle Classification Accuracy', fontsize=12)
    ax2.set_title('Vehicle Classification Accuracy vs Blur Level', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.2, 8.2)
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('classification_accuracy_trends.png', dpi=300, bbox_inches='tight')
    print("Saved accuracy trends to classification_accuracy_trends.png")
    plt.show()

def plot_combined_accuracy_trend(sigma_values, color_accuracies, vehicle_accuracies):
    """Create a single plot comparing both classification types."""
    
    # Sort by sigma values
    sort_indices = np.argsort(sigma_values)
    sigma_sorted = sigma_values[sort_indices]
    color_sorted = color_accuracies[sort_indices]
    vehicle_sorted = vehicle_accuracies[sort_indices]
    
    plt.figure(figsize=(12, 8))
    
    # Plot both lines
    plt.plot(sigma_sorted, color_sorted, 'bo-', linewidth=2, markersize=8, 
            label='Color Classification', alpha=0.8)
    plt.plot(sigma_sorted, vehicle_sorted, 'ro-', linewidth=2, markersize=8, 
            label='Vehicle Classification', alpha=0.8)
    
    plt.xlabel('Blur Sigma (σ)', fontsize=14)
    plt.ylabel('Classification Accuracy', fontsize=14)
    plt.title('Classification Accuracy vs Blur Level', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(-0.2, 8.2)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('combined_classification_accuracy_trends.png', dpi=300, bbox_inches='tight')
    print("Saved combined accuracy trends to combined_classification_accuracy_trends.png")
    plt.show()

def main():
    """Main function to generate accuracy trend plots."""
    print("Loading accuracy data from training_prediction folders...")
    accuracy_data = load_accuracy_data()
    
    if not accuracy_data:
        print("No accuracy data found!")
        return
    
    print(f"\nFound accuracy data for {len(accuracy_data)} blur levels")
    
    # Extract classification accuracies
    sigma_values, color_accuracies, vehicle_accuracies = extract_classification_accuracies(accuracy_data)
    
    if len(sigma_values) == 0:
        print("No valid accuracy data extracted!")
        return
    
    print(f"\nExtracted accuracy data for {len(sigma_values)} sigma levels")
    print(f"Sigma range: {sigma_values.min():.3f} to {sigma_values.max():.3f}")
    
    # Create plots
    plot_accuracy_trends(sigma_values, color_accuracies, vehicle_accuracies)
    plot_combined_accuracy_trend(sigma_values, color_accuracies, vehicle_accuracies)
    
    # Save data to CSV for reference
    df = pd.DataFrame({
        'sigma': sigma_values,
        'color_accuracy': color_accuracies,
        'vehicle_accuracy': vehicle_accuracies
    })
    df_sorted = df.sort_values('sigma')
    df_sorted.to_csv('accuracy_trends_data.csv', index=False)
    print("Saved accuracy data to accuracy_trends_data.csv")

if __name__ == "__main__":
    main()