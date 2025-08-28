#!/bin/bash
# BD3LM Speedrun Training Script
# Optimized for 8xH200 GPUs (single node)

# Set environment variables for optimal performance
# Allow multiple CUDA connections to improve kernel overlap on Hopper
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export PYTORCH_TF32=${PYTORCH_TF32:-1}

# NCCL optimizations for H200
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Prefer NVLink on single node; disable IB by default (override if multi-node)
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
if [ -z "${NCCL_SOCKET_IFNAME}" ]; then
  # try to pick a reasonable default interface (override by exporting)
  IF_CANDIDATE=$(ls /sys/class/net | grep -E 'ib|enp|eno|ens|eth' | head -n1)
  export NCCL_SOCKET_IFNAME=${IF_CANDIDATE}
fi
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_TREE_THRESHOLD=${NCCL_TREE_THRESHOLD:-0}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}

# PyTorch optimizations
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"9.0"}  # H200 architecture

# Training configuration
NUM_GPUS=${NUM_GPUS:-8}
# Positional args: [BLOCK_SIZE] [resume|path]
BLOCK_SIZE_DEFAULT=16
BLOCK_SIZE=${BLOCK_SIZE:-$BLOCK_SIZE_DEFAULT}
RESUME_ARG=""
if [[ -n "${1:-}" ]]; then
  if [[ "$1" =~ ^[0-9]+$ ]]; then
    BLOCK_SIZE=$1
    RESUME_ARG=${2:-}
  else
    RESUME_ARG=$1
  fi
fi
GLOBAL_BATCH=${GLOBAL_BATCH:-1024}
TRAIN_FILES=${TRAIN_FILES:-"./data/fineweb_train_*.bin"}
VAL_FILES=${VAL_FILES:-"./data/fineweb_val_*.bin"}
VAL_TOKENS=${VAL_TOKENS:-10485760}
MAX_STEPS=${MAX_STEPS:-}
TRAIN_TOKENS=${TRAIN_TOKENS:-}
OPTIMIZER=${OPTIMIZER:-mixed}  # mixed | muon | adamw
# Optional AdamW LR override when using adamw or mixed
ADAMW_LR=${ADAMW_LR:-}
EVAL_INTERVAL=${EVAL_INTERVAL:-250}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
# Generate 1 sample per eval by default
SAMPLES_PER_EVAL=${SAMPLES_PER_EVAL:-1}
EVAL_CE_ONLY=${EVAL_CE_ONLY:-1}
COMPILE=${COMPILE:-1}
DDP_FP16_COMPRESS=${DDP_FP16_COMPRESS:-1}
RUN_NAME="bd3lm-speedrun-bs${BLOCK_SIZE}-$(date +%Y%m%d-%H%M%S)"
RESUME_DIR=""
RESUME_PATH=""

# Architecture toggles (override via env)
# Efficient path (two-stream) defaults ON; flex attention stays ON unless explicitly disabled.
USE_SWIGLU=${USE_SWIGLU:-1}
USE_LOCAL_MIXER=${USE_LOCAL_MIXER:-0}
TIE_WEIGHTS=${TIE_WEIGHTS:-1}
USE_FILM=${USE_FILM:-0}
QK_LEARNED_SCALE=${QK_LEARNED_SCALE:-1}
USE_TWO_STREAM=${USE_TWO_STREAM:-1}
NO_FLEX_ATTN=${NO_FLEX_ATTN:-0}
SEDD_MIX_FRAC=${SEDD_MIX_FRAC:-0.05}
RESIDUAL_SCALE=${RESIDUAL_SCALE:-0.7071}  # ~1/sqrt(2)
USE_PRENORM=${USE_PRENORM:-1}
# GQA/MQA and noise replicas
N_KV_HEADS=${N_KV_HEADS:-}
NOISE_REPLICAS=${NOISE_REPLICAS:-}
# Sampler parallel commit
SAMPLE_BLOCK_COMMIT_K=${SAMPLE_BLOCK_COMMIT_K:-}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-}

# If a resume flag/arg is provided, try to reuse the latest checkpoints dir
if [ -n "$RESUME_ARG" ]; then
  case "$RESUME_ARG" in
    latest|--resume|resume)
      # pick latest directory for this block size; fallback to any
      LATEST_DIR=$(ls -1d checkpoints/bd3lm-speedrun-bs${BLOCK_SIZE}-* 2>/dev/null | sort | tail -n1)
      if [ -z "$LATEST_DIR" ]; then
        LATEST_DIR=$(ls -1d checkpoints/* 2>/dev/null | sort | tail -n1)
      fi
      if [ -z "$LATEST_DIR" ]; then
        echo "No checkpoint directories found under checkpoints/. Cannot resume." >&2
        exit 1
      fi
      RUN_NAME=$(basename "$LATEST_DIR")
      RESUME_DIR="$LATEST_DIR"
      ;;
    *)
      # interpret as a run name, directory, or exact ckpt path
      if [ -d "checkpoints/${RESUME_ARG}" ]; then
        RUN_NAME="${RESUME_ARG}"
        RESUME_DIR="checkpoints/${RESUME_ARG}"
      elif [ -d "${RESUME_ARG}" ]; then
        RUN_NAME=$(basename "${RESUME_ARG}")
        RESUME_DIR="${RESUME_ARG}"
      elif [ -f "${RESUME_ARG}" ]; then
        RESUME_PATH="${RESUME_ARG}"
        RUN_NAME=$(basename "$(dirname "${RESUME_ARG}")")
      else
        echo "Resume target not found: ${RESUME_ARG}" >&2
        exit 1
      fi
      ;;
  esac
fi

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
if [ -n "$RESUME_DIR" ]; then echo "  - Resuming from dir: ${RESUME_DIR}"; fi
if [ -n "$RESUME_PATH" ]; then echo "  - Resuming from ckpt: ${RESUME_PATH}"; fi
echo "  - Optimizer: ${OPTIMIZER}"

# Create output directory (ensure existing for resume)
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
    $(if [ -n "${MAX_STEPS}" ]; then echo --max_steps ${MAX_STEPS}; fi) \
    $(if [ -n "${TRAIN_TOKENS}" ]; then echo --train_tokens ${TRAIN_TOKENS}; fi) \
    $(if [ -n "${RESUME_DIR}" ]; then echo --resume_dir "${RESUME_DIR}"; fi) \
    $(if [ -n "${RESUME_PATH}" ]; then echo --resume_path "${RESUME_PATH}"; fi) \
    $(if [ -n "${EVAL_INTERVAL}" ]; then echo --eval_interval ${EVAL_INTERVAL}; fi) \
    $(if [ -n "${SAVE_INTERVAL}" ]; then echo --save_interval ${SAVE_INTERVAL}; fi) \
    $(if [ -n "${SAMPLES_PER_EVAL}" ]; then echo --samples_per_eval ${SAMPLES_PER_EVAL}; fi) \
    $(if [ "${EVAL_CE_ONLY}" = "1" ]; then echo --eval_ce_only; fi) \
    $(if [ -n "${COMPILE}" ]; then echo --compile; fi) \
    $(if [ "${ACT_CKPT}" = "1" ]; then echo --activation_checkpoint; fi) \
    $(if [ "${DDP_FP16_COMPRESS}" = "1" ]; then echo --ddp_fp16_compress; fi) \
    $(if [ "${ENABLE_WANDB}" = "1" ]; then echo --wandb --project_name "${WANDB_PROJECT}"; fi) \
    $(if [ -n "${ADAMW_LR}" ]; then echo --adamw_lr ${ADAMW_LR}; fi) \
    $(if [ "${USE_SWIGLU}" = "1" ]; then echo --use_swiglu; fi) \
    $(if [ "${USE_LOCAL_MIXER}" = "1" ]; then echo --use_local_mixer; fi) \
    $(if [ "${TIE_WEIGHTS}" = "1" ]; then echo --tie_weights; fi) \
    $(if [ "${USE_FILM}" = "1" ]; then echo --use_film; fi) \
    $(if [ "${QK_LEARNED_SCALE}" = "1" ]; then echo --qk_learned_scale; fi) \
    $(if [ "${USE_TWO_STREAM}" = "1" ]; then echo --use_two_stream; fi) \
    $(if [ "${NO_FLEX_ATTN}" = "1" ]; then echo --no_flex_attn; fi) \
    $(if [ -n "${SEDD_MIX_FRAC}" ]; then echo --sedd_mix_frac ${SEDD_MIX_FRAC}; fi) \
    $(if [ -n "${RESIDUAL_SCALE}" ]; then echo --residual_scale ${RESIDUAL_SCALE}; fi) \
    $(if [ "${USE_PRENORM}" != "1" ]; then echo --no_prenorm; fi) \
    $(if [ -n "${N_KV_HEADS}" ]; then echo --n_kv_heads ${N_KV_HEADS}; fi) \
    $(if [ -n "${NOISE_REPLICAS}" ]; then echo --noise_replicas ${NOISE_REPLICAS}; fi) \
    $(if [ -n "${SAMPLE_BLOCK_COMMIT_K}" ]; then echo --sample_block_commit_k ${SAMPLE_BLOCK_COMMIT_K}; fi) \
    $(if [ -n "${GRAD_ACCUM_STEPS}" ]; then echo --grad_accum_steps ${GRAD_ACCUM_STEPS}; fi) \
    2>&1 | tee $(if [ -n "${RESUME_DIR}${RESUME_PATH}" ]; then echo -a; fi) logs/${RUN_NAME}/train.log
