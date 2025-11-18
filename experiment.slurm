#!/bin/bash
#SBATCH --job-name=training_images      # Job name
#SBATCH --output=logs/prediction_%j.out  # Standard output log (%j = job ID)
#SBATCH --error=logs/prediction_%j.err   # Standard error log
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # CPU cores per task
#SBATCH --mem=32G                        # Memory per node
#SBATCH --time=01:00:00                  # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH -A cmda4864fall2025              # Account name

# Print job information
echo "========================================================================"
echo "SLURM Job Information"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "========================================================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results

# Load required modules
echo "Loading modules..."
module load PyTorch
module load scikit-learn/1.3.1
module load JupyterLab

# Install required Python packages
echo ""
echo "Installing required packages..."
pip install --quiet torchvision==0.16.2
pip install --quiet scikit-image
pip install --quiet matplotlib

echo ""
echo "Loaded modules:"
module list
echo ""

# Check Python and PyTorch setup
echo "========================================================================"
echo "Environment Information"
echo "========================================================================"
echo "Python version:"
python --version
echo ""

echo "PyTorch and CUDA information:"
python -c "
import torch
import numpy as np
import sklearn
from skimage import __version__ as skimage_version
import matplotlib

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
print(f'NumPy version: {np.__version__}')
print(f'scikit-learn version: {sklearn.__version__}')
print(f'scikit-image version: {skimage_version}')
print(f'matplotlib version: {matplotlib.__version__}')
"
echo "========================================================================"
echo ""

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# CRITICAL: Set number of workers to match allocated CPUs
# Leave one CPU for main process
NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 1))
if [ $NUM_WORKERS -lt 1 ]; then
    NUM_WORKERS=1
fi

# Configuration - ADJUST THESE PATHS FOR YOUR SETUP
BLUR_LEVEL="Blur4"
CSV_PATH="./dataset_v2/train.csv"
IMAGE_PATH="./blurred_train_images/${BLUR_LEVEL}"
MODEL_PATH="./models/100epochs-CPAB-bceloss-lr0.01-momentum0.9.pt"
OUTPUT_DIR="./results/run_${SLURM_JOB_ID}"
BATCH_SIZE=64
THRESHOLD=0.5

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "Prediction Configuration"
echo "========================================================================"
echo "Blur Level:     $BLUR_LEVEL"
echo "CSV Path:       $CSV_PATH"
echo "Image Path:     $IMAGE_PATH"
echo "Model Path:     $MODEL_PATH"
echo "Output Dir:     $OUTPUT_DIR"
echo "Batch Size:     $BATCH_SIZE"
echo "Threshold:      $THRESHOLD"
echo "Num Workers:    $NUM_WORKERS"
echo "========================================================================"
echo ""

# Verify input files exist
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: CSV file not found: $CSV_PATH"
    exit 1
fi

if [ ! -d "$IMAGE_PATH" ]; then
    echo "ERROR: Image directory not found: $IMAGE_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "✓ All input files verified"
echo ""

# Run the prediction script
echo "========================================================================"
echo "Starting Prediction Script"
echo "========================================================================"
echo ""

python run_prediction.py \
    --csv_path "$CSV_PATH" \
    --image_path "$IMAGE_PATH" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --threshold $THRESHOLD \
    --num_workers $NUM_WORKERS \
    --num_visualize 5 \
    --save_predictions \
    --plot_history

# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================================================"

# Check if the script completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Prediction completed successfully!"
    echo ""
    echo "Results location: $OUTPUT_DIR"
    echo "  - predictions.json"
    echo "  - metrics.json"
    echo "  - predictions_visualization.png"
    echo "  - training_history.png (if available)"
else
    echo "✗ Prediction failed with error code $EXIT_CODE"
    echo ""
    echo "Check the error log: logs/prediction_${SLURM_JOB_ID}.err"
    exit $EXIT_CODE
fi

echo "========================================================================"
echo ""
echo "End Time: $(date)"
echo "Job Duration: $SECONDS seconds"
echo "Job finished!"
echo ""
echo "========================================================================"

# Optional: Copy results to a permanent location
# Uncomment and adjust the path as needed
# FINAL_DEST="/path/to/permanent/storage/results_${SLURM_JOB_ID}"
# echo "Copying results to: $FINAL_DEST"
# mkdir -p "$FINAL_DEST"
# cp -r "$OUTPUT_DIR"/* "$FINAL_DEST/"
# echo "✓ Results copied"

# Optional: Clean up if successful
# rm -rf "$OUTPUT_DIR"