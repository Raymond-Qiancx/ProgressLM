#!/bin/bash

set -x

################################################################################
# EasyR1 Multi-Node Training Script
#
# Usage:
#   Node 1 (Head):   bash run_grpo_3b_multinode.sh head
#   Node 2 (Worker): bash run_grpo_3b_multinode.sh worker <HEAD_NODE_IP>
#
# Example:
#   # On node qgpu01 (head):
#   bash /projects/p32958/chengxuan/ProgressLM/EasyR1/progresslm/run_grpo_3b_multinode.sh head
#
#   # On node qgpu02 (worker): 
#   source /projects/p32958/miniconda3/bin/activate
#   bash /projects/p32958/chengxuan/ProgressLM/EasyR1/progresslm/run_grpo_3b_multinode.sh worker 10.0.0.1
#   bash /projects/p32958/chengxuan/ProgressLM/EasyR1/progresslm/run_grpo_3b_multinode.sh worker 172.20.213.15
################################################################################

# ===== Parse arguments =====
MODE=$1
HEAD_NODE_IP=$2
RAY_PORT=6379

if [ -z "$MODE" ]; then
    echo "Usage: $0 <head|worker> [HEAD_NODE_IP]"
    echo "  head:   Start as head node (runs training)"
    echo "  worker: Start as worker node (joins cluster)"
    exit 1
fi

if [ "$MODE" = "worker" ] && [ -z "$HEAD_NODE_IP" ]; then
    echo "Error: Worker mode requires HEAD_NODE_IP"
    echo "Usage: $0 worker <HEAD_NODE_IP>"
    exit 1
fi

# ===== Model path =====
MODEL_PATH="/projects/p32958/Results/sft_model/qwen25vl_3b_think_sft"

# ===== Cluster config =====
NNODES=4
N_GPUS_PER_NODE=4
REQUIRED_GPUS=$((NNODES * N_GPUS_PER_NODE))

# Timestamp
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# ===== W&B settings =====
export WANDB_API_KEY="ac3c3d795e02ca8885235198ec9a222725622805"
export WANDB_PROJECT="progresslm_grpo"
export WANDB_RUN_GROUP="qwen2_5_vl_3b_progresslm_multinode"
export WANDB_NAME="visual_demo_qwen2p5vl3b_multinode_${TIMESTAMP}"
export WANDB_MODE="online"
export WANDB_DIR="/projects/p32958/Results/wandb_logs"

# ===== Cache directories (avoid disk quota issues) =====
CACHE_ROOT="/gpfs/projects/p32958/chengxuan/.cache"

# HuggingFace cache
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export HF_HUB_CACHE="$CACHE_ROOT/huggingface/hub"

# PyTorch cache
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch/extensions"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torch/inductor"

# Triton cache
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"

# Ray cache - MUST use local /tmp to avoid shared filesystem socket issues
export RAY_TMPDIR="/tmp/ray_${USER}"
export RAY_SESSION_DIR="/tmp/ray_${USER}/session"
export RAY_LOG_DIR="/tmp/ray_${USER}/logs"

# Python bytecode cache
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"

# XDG cache
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

# Temp directories
export TMPDIR="$CACHE_ROOT/tmp"
export TEMP="$CACHE_ROOT/tmp"
export TMP="$CACHE_ROOT/tmp"

# Create all directories
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"
mkdir -p "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$RAY_TMPDIR" "$RAY_SESSION_DIR" "$RAY_LOG_DIR"
mkdir -p "$PYTHONPYCACHEPREFIX" "$XDG_CACHE_HOME" "$TMPDIR"

# ===== NCCL settings for multi-node communication =====
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
# Uncomment and modify if needed:
# export NCCL_SOCKET_IFNAME=ib0  # or eth0, depending on your cluster network

# ===== DataLoader: disable multiprocessing for multi-node =====
export DATALOADER_NUM_WORKERS=0

# ===== W&B: new run, no resume =====
unset WANDB_RUN_ID
unset WANDB_RESUME

echo "=========================================="
echo "Multi-Node Training Script"
echo "Mode: $MODE"
echo "Timestamp: $TIMESTAMP"
if [ "$MODE" = "worker" ]; then
    echo "Head Node IP: $HEAD_NODE_IP"
fi
echo "=========================================="

# ===== Training config =====
# CHECKPOINT_DIR="/projects/p32958/Results/rl_ckpt/qwen25_vl_3b_rl_multinode_${TIMESTAMP}"
CHECKPOINT_DIR="/projects/p32958/Results/rl_ckpt/qwen25_vl_3b_rl_multinode_20251211-064818"

# Get current node IP
CURRENT_IP=$(hostname -I | awk '{print $1}')
echo "Current node IP: $CURRENT_IP"

# ===== Start Ray based on mode =====
if [ "$MODE" = "head" ]; then
    echo "Starting Ray head node..."

    # Stop any existing Ray processes
    ray stop --force 2>/dev/null || true
    sleep 2

    # Start Ray head
    ray start --head --port=$RAY_PORT --num-gpus=4

    echo "=========================================="
    echo "Ray head started at: ${CURRENT_IP}:${RAY_PORT}"
    echo "Workers should connect using:"
    echo "  bash run_grpo_3b_multinode.sh worker ${CURRENT_IP}"
    echo "=========================================="

    # Wait for all workers to join (dynamic wait)
    MAX_WAIT=300      # 5 minutes max wait
    WAIT_INTERVAL=10  # Check every 10 seconds
    WAITED=0

    echo "Waiting for $REQUIRED_GPUS GPUs to join the cluster..."
    while [ $WAITED -lt $MAX_WAIT ]; do
        # Get available GPU count using Python Ray API (more reliable)
        AVAILABLE_GPUS=$(python3 -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null)
        AVAILABLE_GPUS=${AVAILABLE_GPUS:-0}

        echo "[${WAITED}s] Available GPUs: ${AVAILABLE_GPUS}/${REQUIRED_GPUS}"

        if [ "$AVAILABLE_GPUS" -ge "$REQUIRED_GPUS" ]; then
            echo "All GPUs ready!"
            break
        fi

        sleep $WAIT_INTERVAL
        WAITED=$((WAITED + WAIT_INTERVAL))
    done

    if [ "$AVAILABLE_GPUS" -lt "$REQUIRED_GPUS" ]; then
        echo "ERROR: Only $AVAILABLE_GPUS GPUs available after ${MAX_WAIT}s wait."
        echo "Expected $REQUIRED_GPUS GPUs. Aborting."
        ray status
        exit 1
    fi

    # Check Ray cluster status
    ray status

    # Start training
    echo "Starting training..."
    python3 -m verl.trainer.main \
        config=progresslm/configs/multinodes.yaml \
        worker.actor.model.model_path="${MODEL_PATH}" \
        worker.actor.model.tokenizer_path="${MODEL_PATH}" \
        trainer.save_checkpoint_path="${CHECKPOINT_DIR}" \
        trainer.experiment_name="qwen2_5vl3b_grpo_multinode_${TIMESTAMP}" \
        trainer.nnodes=${NNODES} \
        trainer.n_gpus_per_node=${N_GPUS_PER_NODE}

elif [ "$MODE" = "worker" ]; then
    echo "Starting Ray worker node..."

    # Stop any existing Ray processes
    ray stop --force 2>/dev/null || true
    sleep 2

    # Join Ray cluster
    ray start --address="${HEAD_NODE_IP}:${RAY_PORT}" --num-gpus=4

    echo "=========================================="
    echo "Joined Ray cluster at: ${HEAD_NODE_IP}:${RAY_PORT}"
    echo "Worker node is ready. Training will be coordinated by head node."
    echo "=========================================="

    # Keep the worker alive
    echo "Worker running... Press Ctrl+C to stop."
    while true; do
        sleep 60
        ray status 2>/dev/null || {
            echo "Lost connection to Ray cluster. Exiting."
            exit 1
        }
    done
else
    echo "Error: Invalid mode '$MODE'. Use 'head' or 'worker'."
    exit 1
fi
