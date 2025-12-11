"""
Modern Image Classification Prediction Module
"""
# IMPORTS
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Images and Data
import os
import json
import numpy as np
import pandas as pd
import random
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from skimage.transform import resize

# CLASSES
class FineGrainedDataset(Dataset):
    """
    Dataset class for fine-grained image classification.
    
    CRITICAL PREPROCESSING PIPELINE (matches CPAB model exactly):
    1. Load image as PIL Image
    2. Convert RGBA to RGB if needed
    3. Convert to numpy array and normalize by dividing by 255.0
    4. Resize using skimage.transform.resize with mode='reflect', anti_aliasing=True
    5. Take only first 3 channels [:, :, :3]
    6. Convert to torch tensor with float dtype
    7. In dataloader: permute from (H, W, C) to (C, H, W)
    
    NO ImageNet normalization applied - range is [0, 1]
    """

    def __init__(self, csv_path: str, picture_path: str, picturesize: int = 224, transform: bool = True):
        self.transform = transform
        self.picturesize = picturesize
        self.picture_path = Path(picture_path)

        # Load dataframe with one-hot encoding
        self.df = pd.get_dummies(pd.read_csv(csv_path))
        self.df['id'] = (self.df['image_id'].astype(str) + "_" + 
                         self.df['tag_id'].astype(str) + ".png")
        self.df = self.df.iloc[:, 10:]  # keep only one-hot class columns + id
        self.df = self.df.replace(-1, 0)

    def __getitem__(self, idx):
        sample_id = self.df['id'].iloc[idx]
        targets = self.df.loc[idx, self.df.columns != 'id'].values.astype('float32')

        # Load image EXACTLY as original
        img_path = self.picture_path / sample_id
        image = np.asarray(Image.open(img_path))

        if self.transform:
            # CPAB preprocessing pipeline - EXACTLY matching original
            # CRITICAL: Do NOT change the order or any detail!
            
            # Step 1: Copy image
            img = image
            img_normalized = np.copy(img)
            
            # Step 2: Normalize (use 255. not 255.0 to match original exactly)
            img_normalized = img_normalized / 255.
            
            # Step 3: Resize with specific parameters matching CPAB
            img_resized = resize(
                img_normalized,
                output_shape=(self.picturesize, self.picturesize),
                mode='reflect',
                anti_aliasing=True
            )[:, :, :3]  # Take only first 3 channels
            
            # Step 4: Convert to tensor WITHOUT .float() initially
            # Original uses torch.from_numpy which preserves float64 from resize
            sample = {
                'id': sample_id,
                'targets': torch.from_numpy(targets),
                'image': torch.from_numpy(img_resized)
            }
        else:
            sample = {
                'id': sample_id,
                'targets': targets,
                'image': image
            }

        return sample

    def __len__(self):
        return len(self.df)


# Legacy model classes for backward compatibility with old saved models
class Head(nn.Module):
    """Legacy Head class from original CPAB implementation."""
    def __init__(self):
        super(Head, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 37)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class Net(nn.Module):
    """Legacy Net class from original CPAB implementation."""
    def __init__(self, num_classes=37):
        super(Net, self).__init__()
        
        # Load pretrained ResNet50 (using old API for compatibility)
        self.model_conv = models.resnet50(pretrained=False)
        
        # Disable autograd for resnet
        for param in self.model_conv.parameters():
            param.requires_grad = False
        
        # Change fully connected layer
        num_ftrs = self.model_conv.fc.in_features
        self.model_conv.fc = nn.Linear(num_ftrs, 1024)
        
        # Output layer
        self.output = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.model_conv(x)
        x = F.relu(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x


class FineGrainedClassifier(nn.Module):
    """Neural network for fine-grained classification matching CPAB architecture."""
    
    def __init__(self, num_classes: int, feature_extract: bool = True):
        """
        Args:
            num_classes: Number of output classes
            feature_extract: If True, freeze ResNet weights
        """
        super(FineGrainedClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze ResNet parameters if feature extracting
        if feature_extract:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Replace final layer to match CPAB architecture
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1024)
        
        # Output layer
        self.output = nn.Linear(1024, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = F.relu(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x


class Predictor:
    """Prediction wrapper with preprocessing matching CPAB augmentation."""

    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()
        
        # Get class names (excluding 'id' column)
        self.class_names = [col for col in self.dataset.df.columns if col != 'id']

    def predict_batch(self, dataloader, threshold=0.5):
        """Run predictions on entire dataloader (legacy method - no scores)."""
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                # CRITICAL: Permute from (B, H, W, C) to (B, C, H, W)
                inputs = batch['image'].permute(0, 3, 1, 2).to(self.device, dtype=torch.float)
                labels = batch['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(inputs)

                for pred, label in zip(outputs, labels):
                    pred_classes = [self.class_names[i] 
                                    for i, p in enumerate(pred) 
                                    if p > threshold]
                    true_classes = [self.class_names[i] 
                                    for i, l in enumerate(label) 
                                    if l > threshold]
                    predictions.append(pred_classes)
                    true_labels.append(true_classes)

        return predictions, true_labels
    
    def predict_batch_with_scores(self, dataloader, threshold=0.5):
        """
        Run predictions on entire dataloader with scores.
        
        Returns:
            List of dicts with format:
            {
                'id': image_id,
                'pred_scores': {class_name: score, ...},
                'pred': [predicted_classes],
                'true': [true_classes]
            }
        """
        results = []

        with torch.no_grad():
            for batch in dataloader:
                # CRITICAL: Permute from (B, H, W, C) to (B, C, H, W)
                inputs = batch['image'].permute(0, 3, 1, 2).to(self.device, dtype=torch.float)
                labels = batch['targets'].to(self.device, dtype=torch.float)
                ids = batch['id']

                outputs = self.model(inputs)

                for sample_id, pred, label in zip(ids, outputs, labels):
                    # Convert tensors to CPU and numpy
                    pred_np = pred.cpu().numpy()
                    label_np = label.cpu().numpy()
                    
                    # Create score dictionary for all classes
                    pred_scores = {
                        self.class_names[i]: float(pred_np[i])
                        for i in range(len(self.class_names))
                    }
                    
                    # Get predicted classes above threshold
                    pred_classes = [
                        self.class_names[i] 
                        for i in range(len(self.class_names))
                        if pred_np[i] > threshold
                    ]
                    
                    # Get true classes
                    true_classes = [
                        self.class_names[i]
                        for i in range(len(self.class_names))
                        if label_np[i] > threshold
                    ]
                    
                    results.append({
                        'id': sample_id,
                        'pred_scores': pred_scores,
                        'pred': pred_classes,
                        'true': true_classes
                    })

        return results
    
    def save_predictions_json(self, dataloader, output_path, threshold=0.5, include_scores=True):
        """
        Save predictions to JSON file.
        
        Args:
            dataloader: DataLoader to run predictions on
            output_path: Path to save JSON file
            threshold: Classification threshold
            include_scores: If True, include all class scores in output
        """
        print(f"\nGenerating predictions{'with scores' if include_scores else ''}...")
        
        if include_scores:
            results = self.predict_batch_with_scores(dataloader, threshold)
        else:
            # Legacy format without scores
            predictions, true_labels = self.predict_batch(dataloader, threshold)
            results = []
            for i, (pred, true) in enumerate(zip(predictions, true_labels)):
                results.append({
                    'id': self.dataset.df['id'].iloc[i],
                    'pred': pred,
                    'true': true
                })
        
        print(f"Saving {len(results)} predictions to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Predictions saved successfully")
        
        # Print sample
        if results:
            print("\nSample prediction:")
            sample = results[0].copy()
            # Truncate scores for display if present
            if 'pred_scores' in sample and len(sample['pred_scores']) > 5:
                displayed_scores = dict(list(sample['pred_scores'].items())[:5])
                sample['pred_scores'] = {**displayed_scores, '...': f'{len(sample["pred_scores"]) - 5} more classes'}
            print(json.dumps(sample, indent=2))
    
    def visualize_predictions(self, dataloader, num_samples=3, threshold=0.5):
        """Visualize predictions on sample images."""
        batch = next(iter(dataloader))
        inputs = batch['image'].permute(0, 3, 1, 2).to(self.device, dtype=torch.float)
        labels = batch['targets'].to(self.device, dtype=torch.float)

        with torch.no_grad():
            outputs = self.model(inputs)

        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for i, (pred, label, img) in enumerate(zip(outputs[:num_samples], labels[:num_samples], inputs[:num_samples])):
            pred_classes = [self.class_names[j] for j, p in enumerate(pred) 
                           if p > threshold]
            true_classes = [self.class_names[j] for j, l in enumerate(label) 
                           if l > threshold]

            img_display = img.cpu().numpy().transpose(1, 2, 0)
            axes[i].imshow(img_display)
            axes[i].axis('off')
            axes[i].set_title(f"Predicted: {', '.join(pred_classes)}\nGround Truth: {', '.join(true_classes)}", 
                            fontsize=10)

        plt.tight_layout()
        plt.show()


# FUNCTIONS
def fix_model_compatibility(model):
    """Fix compatibility issues with models from different PyTorch versions."""
    # Fix AvgPool2d divisor_override issue
    for module in model.modules():
        if isinstance(module, nn.AvgPool2d):
            if not hasattr(module, 'divisor_override'):
                module.divisor_override = None
    return model


def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Load a trained model with compatibility for old and new formats.
    Handles legacy models saved as full objects vs state dicts.
    """
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Case 1: model saved as state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        print("  Format: state_dict (wrapped)")
        model = FineGrainedClassifier(num_classes=num_classes, feature_extract=False)
        model.load_state_dict(checkpoint['state_dict'])
    
    # Case 2: pure state dict (no wrapper)
    elif isinstance(checkpoint, dict):
        print("  Format: state_dict (direct)")
        model = FineGrainedClassifier(num_classes=num_classes, feature_extract=False)
        model.load_state_dict(checkpoint)
    
    # Case 3: entire model object (legacy - CPAB models use this)
    else:
        print(f"  Format: Full model object (type: {type(checkpoint).__name__})")
        model = checkpoint
        
        # Verify model has required attributes
        if not hasattr(model, 'forward'):
            raise AttributeError("Loaded model missing 'forward' method")

    model = fix_model_compatibility(model)
    model = model.to(device)
    model.eval()
    print("  Model loaded successfully")
    return model


def plot_training_history(model_path: str):
    """
    Plot training history from saved CSV files.
    
    Args:
        model_path: Base path for model and CSV files (without extension)
    """
    losses_train = pd.read_csv(f"{model_path}Loss_train.csv")
    losses_valid = pd.read_csv(f"{model_path}Loss_valid.csv")
    accuracy = pd.read_csv(f"{model_path}MAP.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(losses_train['Epoch'], losses_train['Loss'], '-b', label='Train', linewidth=2)
    ax1.plot(losses_valid['Epoch'], losses_valid['Loss'], '-r', label='Valid', linewidth=2)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # MAP plot
    ax2.plot(accuracy['Epoch'], accuracy['MAP'], '-b', linewidth=2)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('MAP', fontsize=12)
    ax2.set_title('Mean Average Precision', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print final metrics
    print("\n" + "="*50)
    print("TRAINING HISTORY SUMMARY")
    print("="*50)
    print(f"Final Training Loss:   {losses_train['Loss'].iloc[-1]:.4f}")
    print(f"Final Validation Loss: {losses_valid['Loss'].iloc[-1]:.4f}")
    print(f"Final MAP:             {accuracy['MAP'].iloc[-1]:.4f}")
    print("="*50 + "\n")


# Only run in interactive session
if __name__ == "__main__":
    # Configuration
    CSV_PATH = './dataset_v2/train.csv'
    IMAGE_PATH = './dataset_v2/root/train/cropped/'
    MODEL_PATH = './models/100epochs-customloss-lr0.002-momentum0.9.pt'
    OUTPUT_JSON = './predictions_with_scores.json'
    BATCH_SIZE = 64
    THRESHOLD = 0.5
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = FineGrainedDataset(
        csv_path=CSV_PATH,
        picture_path=IMAGE_PATH,
        picturesize=224,
        transform=True
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    num_classes = dataset.df.shape[1] - 1  # Exclude 'id' column
    model = load_model(MODEL_PATH, num_classes, device)
    
    # Create predictor
    predictor = Predictor(model, dataset, device)
    
    # Save predictions with scores to JSON
    predictor.save_predictions_json(
        dataloader, 
        OUTPUT_JSON, 
        threshold=THRESHOLD,
        include_scores=True
    )
    
    # Visualize predictions
    print("\nVisualizing predictions...")
    predictor.visualize_predictions(dataloader, num_samples=3, threshold=THRESHOLD)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history('./models/100epochs-customloss-lr0.002-momentum0.9')