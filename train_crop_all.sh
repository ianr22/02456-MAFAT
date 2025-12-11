#!/bin/bash
#SBATCH --job-name=crop_images      # Job name
#SBATCH --output=logs/cropping/prediction_%j.out  # Standard output log (%j = job ID)
#SBATCH --error=logs/cropping/prediction_%j.err   # Standard error log
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # CPU cores per task
#SBATCH --mem=32G                        # Memory per node
#SBATCH --time=01:00:00                  # Time limit hrs:min:sec
#SBATCH -A cmda4864fall2025              # Account name
#SBATCH --gres=gpu:1                     # Number of GPUs

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

# CONFIG
BASE_IMAGE_DIR="./blurred_train_images_demo"
CROP_LOGS="./logs/cropping"

# Create log directory
mkdir -p "$CROP_LOGS"

# Load modules
module load SciPy-bundle
module load Pillow

pip install opencv-python
pip install tqdm

# Crop images
python3 train_crop_all.py
