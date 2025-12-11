#!/bin/bash

###############################################################################
# Submit SLURM jobs for all blur levels
###############################################################################

echo "========================================================================"
echo "Submitting Prediction Jobs for All Blur Levels"
echo "========================================================================"
echo "Start Time: $(date)"
echo ""

# Configuration
BASE_IMAGE_DIR="./blurred_train_images_demo"
CSV_PATH="./dataset_v2/train_balanced_general_class.csv"
MODEL_PATH="./models/100epochs-CPAB-bceloss-lr0.01-momentum0.9.pt"
BASE_OUTPUT_DIR="./results/demo"
BATCH_SIZE=64
THRESHOLD=0.5
INCLUDE_SCORES=true
NUM_BLUR_LEVELS=2

# Create necessary directories
mkdir -p logs
mkdir -p "$BASE_OUTPUT_DIR"

# Verify base directories exist
if [ ! -d "$BASE_IMAGE_DIR" ]; then
    echo "ERROR: Base image directory not found: $BASE_IMAGE_DIR"
    exit 1
fi

if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: CSV file not found: $CSV_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "Base directories verified"
echo ""

# Array to store job IDs
declare -a JOB_IDS

# Loop through blur levels
for BLUR_LEVEL in $(seq 0 $((NUM_BLUR_LEVELS - 1))); do
    IMAGE_PATH="${BASE_IMAGE_DIR}/blur_${BLUR_LEVEL}"
    
    # Check if this blur level directory exists
    if [ ! -d "$IMAGE_PATH" ]; then
        echo "WARNING: Skipping blur_${BLUR_LEVEL} - directory not found: $IMAGE_PATH"
        continue
    fi
    
    # Count images in directory
    NUM_IMAGES=$(find "$IMAGE_PATH" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
    echo "Processing blur_${BLUR_LEVEL}:"
    echo "  Directory: $IMAGE_PATH"
    echo "  Images: $NUM_IMAGES"
    
    if [ "$NUM_IMAGES" -eq 0 ]; then
        echo "  WARNING: No images found, skipping"
        echo ""
        continue
    fi
    
    # Build sbatch command with environment variables
    SBATCH_CMD="sbatch \
        --export=ALL,BLUR_LEVEL=blur_${BLUR_LEVEL},CSV_PATH=${CSV_PATH},IMAGE_PATH=${IMAGE_PATH},MODEL_PATH=${MODEL_PATH},BATCH_SIZE=${BATCH_SIZE},THRESHOLD=${THRESHOLD},INCLUDE_SCORES=${INCLUDE_SCORES} \
        --job-name=pred_blur${BLUR_LEVEL} \
        --output=logs/prediction_blur${BLUR_LEVEL}_%j.out \
        --error=logs/prediction_blur${BLUR_LEVEL}_%j.err \
        experiment.slurm"
    
    # Submit job
    JOB_OUTPUT=$($SBATCH_CMD)
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+')
    
    if [ -n "$JOB_ID" ]; then
        JOB_IDS+=("$JOB_ID")
        echo "  Submitted job: $JOB_ID"
    else
        echo "  ERROR: Failed to submit job for blur_${BLUR_LEVEL}"
    fi
    
    echo ""
    
    # Optional: Add small delay between submissions to avoid overwhelming scheduler
    sleep 0.5
done

echo "========================================================================"
echo "Submission Summary"
echo "========================================================================"
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo ""
echo "Job IDs:"
for JOB_ID in "${JOB_IDS[@]}"; do
    echo "  - $JOB_ID"
done
echo ""

# Print monitoring commands
echo "========================================================================"
echo "Monitoring Commands"
echo "========================================================================"
echo "Check queue status:"
echo "  squeue -u \$USER"
echo ""
echo "Check specific jobs:"
echo "  squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo ""
echo "Cancel all jobs:"
echo "  scancel $(IFS=' '; echo "${JOB_IDS[*]}")"
echo ""
echo "Monitor output logs:"
echo "  tail -f logs/prediction_blur*_*.out"
echo ""
echo "Check job status:"
echo "  sacct -j $(IFS=,; echo "${JOB_IDS[*]}") --format=JobID,JobName,State,ExitCode,Elapsed"
echo "========================================================================"
echo ""

# Optional: Wait for all jobs to complete
read -p "Wait for all jobs to complete? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Waiting for jobs to complete..."
    echo "(Press Ctrl+C to stop waiting and return to terminal)"
    echo ""
    
    for JOB_ID in "${JOB_IDS[@]}"; do
        echo "Waiting for job $JOB_ID..."
        while squeue -j "$JOB_ID" 2>/dev/null | grep -q "$JOB_ID"; do
            sleep 10
        done
        echo "  Job $JOB_ID completed"
    done
    
    echo ""
    echo "All jobs completed!"
    echo ""
    
    # Check job statuses
    echo "Final job statuses:"
    sacct -j $(IFS=,; echo "${JOB_IDS[*]}") --format=JobID,JobName,State,ExitCode,Elapsed
    echo ""
    
    # List results
    echo "Results locations:"
    for BLUR_LEVEL in $(seq 0 $((NUM_BLUR_LEVELS - 1))); do
        RESULT_PATTERN="${BASE_OUTPUT_DIR}/run_*/predictions*blur_${BLUR_LEVEL}*"
        if ls $RESULT_PATTERN 2>/dev/null; then
            echo "  blur_${BLUR_LEVEL}: Found results"
        fi
    done
fi

echo ""
echo "Script completed at: $(date)"
echo "========================================================================"