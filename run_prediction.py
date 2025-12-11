"""
Command-line interface for running predictions on HPC cluster
"""
# IMPORTS
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import json
from pathlib import Path
from torch.utils.data import DataLoader
import sys

# From experiment_runner module
from experiment_runner import (
    FineGrainedDataset,
    FineGrainedClassifier,
    Predictor,
    load_model,
    plot_training_history,
    Head,  # Legacy class needed for unpickling old models
    Net    # Legacy class needed for unpickling old models
)

# FUNCTIONS 
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run image classification predictions on HPC cluster with CPAB preprocessing'
    )
    
    # Required arguments
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to CSV file with labels'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to directory containing images'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model file (.pt)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save results (default: ./results)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for predictions (default: 64)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=224,
        help='Image size for resizing (default: 224 - MUST match CPAB training)'
    )
    parser.add_argument(
        '--num_visualize',
        type=int,
        default=5,
        help='Number of samples to visualize (default: 5, set to 0 to disable)'
    )
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='Save predictions to JSON file'
    )
    parser.add_argument(
        '--include_scores',
        action='store_true',
        help='Include prediction scores for all classes in JSON output'
    )
    parser.add_argument(
        '--plot_history',
        action='store_true',
        help='Plot training history (requires loss CSV files)'
    )
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    
    return parser.parse_args()


def setup_output_dir(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")
    return output_path


def calculate_metrics(results):
    """
    Calculate and display prediction metrics from results with scores.
    
    Args:
        results: List of dicts with 'pred' and 'true' keys
    """
    total_samples = len(results)
    correct_predictions = 0
    partial_correct = 0
    
    # Calculate per-sample metrics
    for result in results:
        pred = result['pred']
        label = result['true']
        
        if set(pred) == set(label):
            correct_predictions += 1
        elif len(set(pred) & set(label)) > 0:
            partial_correct += 1
    
    exact_accuracy = (correct_predictions / total_samples) * 100
    partial_accuracy = ((correct_predictions + partial_correct) / total_samples) * 100
    
    print("\n" + "="*60)
    print("PREDICTION METRICS")
    print("="*60)
    print(f"Total samples:     {total_samples}")
    print(f"Exact matches:     {correct_predictions} ({exact_accuracy:.2f}%)")
    print(f"Partial matches:   {partial_correct} ({(partial_correct/total_samples)*100:.2f}%)")
    print(f"Partial + Exact:   {correct_predictions + partial_correct} ({partial_accuracy:.2f}%)")
    print(f"No matches:        {total_samples - correct_predictions - partial_correct}")
    print("="*60 + "\n")
    
    return {
        'total': total_samples,
        'exact': correct_predictions,
        'partial': partial_correct,
        'exact_accuracy': exact_accuracy,
        'partial_accuracy': partial_accuracy
    }


def verify_preprocessing():
    """Print preprocessing pipeline being used."""
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE (CPAB Compatible)")
    print("="*60)
    print("1. Load image as numpy array")
    print("2. Normalize: divide by 255.0 → range [0, 1]")
    print("3. Resize: skimage.transform.resize")
    print("   - output_shape=(224, 224)")
    print("   - mode='reflect'")
    print("   - anti_aliasing=True")
    print("4. Take first 3 channels [:, :, :3]")
    print("5. Convert to torch tensor (float32)")
    print("6. Permute: (H, W, C) → (C, H, W)")
    print("7. NO ImageNet normalization applied")
    print("="*60 + "\n")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup
    print("\n" + "="*60)
    print("IMAGE CLASSIFICATION PREDICTION - HPC CLUSTER")
    print("CPAB Model Inference Pipeline")
    print("="*60 + "\n")
    
    # Verify preprocessing
    verify_preprocessing()
    
    # Device setup
    if args.no_cuda:
        device = torch.device("cpu")
        print("CUDA disabled by user, using CPU")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {gpu_mem_gb:.2f} GB")
        else:
            print("CUDA not available, using CPU")
    print()
    
    # Create output directory
    output_path = setup_output_dir(args.output_dir)
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = FineGrainedDataset(
            csv_path=args.csv_path,
            picture_path=args.image_path,
            picturesize=args.image_size,
            transform=True
        )
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Number of classes: {dataset.df.shape[1] - 1}")
        print(f"Image size: {args.image_size}x{args.image_size}")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create dataloader
    print(f"\nCreating dataloader...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Dataloader created ({len(dataloader)} batches)")
    
    # Load model
    print(f"\nLoading model...")
    try:
        num_classes = dataset.df.shape[1] - 1
        model = load_model(args.model_path, num_classes, device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create predictor
    predictor = Predictor(model, dataset, device)
    
    # Run predictions and save if requested
    if args.save_predictions:
        print(f"\nRunning predictions...")
        print(f"  Threshold: {args.threshold}")
        print(f"  Include scores: {args.include_scores}")
        
        try:
            # Determine output filename based on score inclusion
            if args.include_scores:
                output_file = output_path / 'predictions_with_scores.json'
            else:
                output_file = output_path / 'predictions.json'
            
            # Save predictions using the unified method
            predictor.save_predictions_json(
                dataloader, 
                output_file,
                threshold=args.threshold,
                include_scores=args.include_scores
            )
            
            # Load results for metrics calculation
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            print("Predictions complete!")
            
            # Calculate and display metrics
            metrics = calculate_metrics(results)
            
            # Save metrics
            metrics_file = output_path / 'metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to: {metrics_file}")
            
        except Exception as e:
            print(f"ERROR: Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\nSkipping prediction save (use --save_predictions to enable)")
    
    # Visualize predictions
    if args.num_visualize > 0:
        print(f"\nGenerating visualizations for {args.num_visualize} samples...")
        try:
            predictor.visualize_predictions(
                dataloader, 
                num_samples=args.num_visualize,
                threshold=args.threshold
            )
            # Save figure
            import matplotlib.pyplot as plt
            viz_path = output_path / 'predictions_visualization.png'
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {viz_path}")
            plt.close()
        except Exception as e:
            print(f"WARNING: Visualization failed: {e}")
    
    # Plot training history
    if args.plot_history:
        print("\nPlotting training history...")
        try:
            # Remove .pt extension to get base path
            model_base_path = str(Path(args.model_path).with_suffix(''))
            plot_training_history(model_base_path)
            
            import matplotlib.pyplot as plt
            history_path = output_path / 'training_history.png'
            plt.savefig(history_path, dpi=150, bbox_inches='tight')
            print(f"Training history saved to: {history_path}")
            plt.close()
        except FileNotFoundError as e:
            print(f"WARNING: Could not find training history CSV files")
            print(f"   Expected files: {model_base_path}Loss_train.csv, etc.")
        except Exception as e:
            print(f"WARNING: Could not plot training history: {e}")
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE!")
    print(f"Results saved to: {output_path.absolute()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()