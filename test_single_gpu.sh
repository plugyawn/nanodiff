#!/bin/bash
# Single GPU test for BD3LM to verify model works

# Set environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

# Training configuration
BLOCK_SIZE=${1:-16}
RUN_NAME="bd3lm-test-single-gpu-$(date +%Y%m%d-%H%M%S)"

echo "Starting BD3LM Single GPU Test"
echo "Configuration:"
echo "  - GPUs: 1"
echo "  - Block Size: ${BLOCK_SIZE}"
echo "  - Run Name: ${RUN_NAME}"

# Create output directory
mkdir -p checkpoints/${RUN_NAME}
mkdir -p logs/${RUN_NAME}

# Launch single GPU quick test
CUDA_VISIBLE_DEVICES=0 python train_bd3lm.py \
    --block_size ${BLOCK_SIZE} \
    --batch_size 8 \
    --test \
    2>&1 | tee logs/${RUN_NAME}/train.log
