#!/bin/bash

#SBATCH --account=ms-yromanoma42
#SBATCH --nodes=2                         # 2 nodes
#SBATCH --ntasks-per-node=1               # 1 task per node
#SBATCH --gpus-per-node=2                 # 2 GPUs per node
#SBATCH --mem=450G                        # 450GB RAM per node
#SBATCH --output=slurm_logs/%x_%j.out     # Output logs
#SBATCH --error=slurm_logs/%x_%j.err      # Error logs
#SBATCH --time=7-00:00:00                 # 7 days

# --- Configure Environment ---
export HF_HOME=/work/projects/ms-yromanoma42/yromanoma42/hf
export TRITON_CACHE_DIR=/tmp/triton_cache

# --- NCCL Optimizations (InfiniBand) ---
export NCCL_IB_DISABLE=0                  # Enable InfiniBand
export NCCL_IB_HCA=mlx5                   # Mellanox HCA
export NCCL_SOCKET_IFNAME=ib0             # Force IB interface

# --- Distributed Training Setup ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n1)

# --- Load Modules ---
spack load cuda@11.8.0
source rlcard/bin/activate

# --- Run with Torchrun ---
srun --nodes=$SLURM_JOB_NUM_NODES \
     --ntasks-per-node=1 \
     --gpus-per-task=$SLURM_GPUS_PER_NODE \
     torchrun \
     --nnodes=$SLURM_JOB_NUM_NODES \
     --nproc_per_node=$SLURM_GPUS_PER_NODE \
     --rdzv_id=$SLURM_JOB_ID \
     --rdzv_backend=c10d \
     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     ds_rllogging.py \
     --deepspeed_config ds_config.json