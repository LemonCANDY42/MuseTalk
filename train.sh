#!/bin/bash

# MuseTalk Training Script
# This script combines both training stages for the MuseTalk model
# Usage: sh train.sh [stage1|stage2] [true|false]
# Example: sh train.sh stage1 true  # To run stage 1 training with memory profiling
# Example: sh train.sh stage2 false  # To run stage 2 training without memory profiling

# Check if stage argument is provided
if [ $# -ne 1 ]; then
    echo "Error: Please specify the training stage and memory profiling option"
    echo "Usage: ./train.sh [stage1|stage2] [true|false]"
    exit 1
fi

STAGE=$1
USE_SCALENE=$2

# Validate stage argument
if [ "$STAGE" != "stage1" ] && [ "$STAGE" != "stage2" ]; then
    echo "Error: Invalid stage. Must be either 'stage1' or 'stage2'"
    exit 1
fi

# Launch distributed training using accelerate
# --config_file: Path to the GPU configuration file
# --main_process_port: Port number for the main process, used for distributed training communication
# train.py: Training script
# --config: Path to the training configuration file
echo "Starting $STAGE training..."

if [ "$USE_SCALENE" = "true" ]; then
    echo "Using Scalene for memory profiling"
    accelerate launch --config_file ./configs/training/gpu.yaml \
            --main_process_port 29502  \
            --no_python scalene --gpu --memory  --profile-exclude lib/python3 --html --outfile train_profile.html --- train.py --config ./configs/training/$STAGE.yaml
else
    echo "Using Accelerate for training"
    accelerate launch --config_file ./configs/training/gpu.yaml \
                      --main_process_port 29502 \
                      train.py --config ./configs/training/$STAGE.yaml
fi

echo "Training completed for $STAGE" 