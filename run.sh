#!/bin/bash
# BD3LM Speedrun Training Script
# Optimized for 8xH200 GPUs (single node)

# Set environment variables for optimal performance
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

# NCCL optimizations for H200
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
if [ -z "${NCCL_SOCKET_IFNAME}" ]; then
  # try to pick a reasonable default interface (override by exporting)
  IF_CANDIDATE=$(ls /sys/class/net | grep -E 'ib|enp|eno|ens|eth' | head -n1)
  export NCCL_SOCKET_IFNAME=${IF_CANDIDATE}
fi
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_TREE_THRESHOLD=${NCCL_TREE_THRESHOLD:-0}

# PyTorch optimizations
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"9.0"}  # H200 architecture

# Training configuration
NUM_GPUS=${NUM_GPUS:-8}
BLOCK_SIZE=${1:-16}  # Default block size 16
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
TRAIN_FILES=${TRAIN_FILES:-"./data/fineweb_train_*.bin"}
VAL_FILES=${VAL_FILES:-"./data/fineweb_train_*.bin"}
VAL_TOKENS=${VAL_TOKENS:-10485760}
OPTIMIZER=${OPTIMIZER:-mixed}  # mixed | muon | adamw
# Optional AdamW LR override when using adamw or mixed
ADAMW_LR=${ADAMW_LR:-}
RUN_NAME="bd3lm-speedrun-bs${BLOCK_SIZE}-$(date +%Y%m%d-%H%M%S)"

# Weights & Biases
ENABLE_WANDB=${ENABLE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-bd3lm-speedrun}
# Allow user to pre-set WANDB_ENTITY, WANDB_MODE, etc. if desired
export WANDB_PROJECT
export WANDB_NAME="$RUN_NAME"

echo "Starting BD3LM Speedrun Training"
echo "Configuration:"
echo "  - GPUs: ${NUM_GPUS}"
echo "  - Block Size: ${BLOCK_SIZE}"
echo "  - Run Name: ${RUN_NAME}"
echo "  - Optimizer: ${OPTIMIZER}"

# Create output directory
mkdir -p checkpoints/${RUN_NAME}
mkdir -p logs/${RUN_NAME}

# Launch distributed training
torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    train_bd3lm.py \
    --run_name ${RUN_NAME} \
    --block_size ${BLOCK_SIZE} \
    --batch_size ${GLOBAL_BATCH} \
    --train_files "${TRAIN_FILES}" \
    --val_files "${VAL_FILES}" \
    --val_tokens ${VAL_TOKENS} \
    --optimizer ${OPTIMIZER} \
    $(if [ "${ENABLE_WANDB}" = "1" ]; then echo --wandb --project_name "${WANDB_PROJECT}"; fi) \
    $(if [ -n "${ADAMW_LR}" ]; then echo --adamw_lr ${ADAMW_LR}; fi) \
    2>&1 | tee logs/${RUN_NAME}/train.log
