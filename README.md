# Distributed LLM-UNO Sample Repository

This repository contains sample scripts for running a distributed UNO game with an LLM agent using RLCard and Hugging Face models.

## Prerequisites

1. **Python** (>=3.8)
2. **Install from GitHub** (includes all dependencies via `setup.py`):

   ```bash
   pip install git+https://github.com/yagoromano/llm-uno.git
   ```
3. **Hugging Face CLI**:

   ```bash
   pip install huggingface-cli
   huggingface-cli login
   ```
4. **PyTorch** with CUDA support (for GPU inference).
5. **DeepSpeed** (if using DeepSpeed inference in `llm_dist_sample.py`):

   ```bash
   pip install deepspeed
   ```

## Repository Structure

```text
llm_uno/               # Python package with custom game and agents
  ├── custom_uno_game.py
  ├── random_agent.py
  └── llm_dist/         # Distributed LLM agent implementation
      └── dist_ClozeAgent.py

llm_uno_sample.py      # Single-node UNO+LLM example
llm_dist_sample.py     # Multi-node distributed UNO+LLM example
README.md              # This file
ds_config.json         # DeepSpeed config (optional)
llama70B_cloze.txt     # Prompt template for Llama-70B
```

## Configuration

You can choose where to cache Hugging Face models by setting the `HF_HOME` environment variable:

```bash
export HF_HOME=/path/to/your/hf_cache
```

## Running the Single-Node Example

This example runs a single UNO game with one LLM player on a single GPU/node.

```bash
# Ensure HF_HOME is set
export HF_HOME=/path/to/your/hf_cache

# Run the sample with a Hugging Face LLM
# (Uncomment your chosen model block in llm_uno_sample.py; default models are commented)
python3 llm_uno_sample.py

# Or run with OpenRouter agent (one required argument)
python3 llm_uno_sample.py --api_key YOUR_OPENROUTER_API_KEY
```
- By default the script executes the Hugging Face model block. To switch models, uncomment the appropriate lines in `llm_uno_sample.py`.
- When using OpenRouter, only the `--api_key` flag is required; model ID and template path use the built-in defaults.
- You can change the model by modifying the `model_id` in `llm_uno_sample.py`. If you switch to a smaller model, adjust prompt tags/templates in `llama8B_cloze.txt` accordingly.

## Running the Multi-Node Example (Large Models)

For large models (e.g., LLaMA-70B) that require multiple GPUs/nodes, use `llm_dist_sample.py` with `torchrun` under SLURM.

```bash
# High-speed interconnect settings (InfiniBand)
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0

# Triton cache (optional)
export TRITON_CACHE_DIR=/tmp/triton_cache

# Hugging Face cache
export HF_HOME=/path/to/your/hf_cache

# SLURM rendezvous settings
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$(comm -23 <(seq 49152 65535 | sort) \
  <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n1)

# Load CUDA with Spack (if needed)
spack load cuda@11.8.0

# Activate your Python environment
source rlcard/bin/activate

# Launch distributed run
srun --nodes=$SLURM_JOB_NUM_NODES \
     --ntasks-per-node=1 \
     --gpus-per-task=$SLURM_GPUS_PER_NODE \
     torchrun \
       --nnodes=$SLURM_JOB_NUM_NODES \
       --nproc_per_node=$SLURM_GPUS_PER_NODE \
       --rdzv_id=$SLURM_JOB_ID \
       --rdzv_backend=c10d \
       --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     llm_dist_sample.py \
```
* You can swap in a different script or model by editing `llm_dist_sample.py`.

## Notes

* Ensure that `ds_config.json` (DeepSpeed config) is present if you specify `--deepspeed_config`.
* Prompt templates and agent parameters should match your chosen model’s token requirements.

---

Happy gaming and experimentation with distributed LLM agents in UNO!
