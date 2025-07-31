# Distributed LLM-UNO Sample Repository

This repository contains sample scripts for running a distributed UNO game with an LLM agent using RLCard, Hugging Face models, and optionally OpenRouter.

## Installation

Make sure that you have **Python 3.8+** and pip installed.

### Install from GitHub (base package)
```bash
pip install "llm-uno @ git+https://github.com/yagoromano/llm-uno.git"

```

### (Optional) Install with distributed extra
```bash
pip install "llm-uno[distributed] @ git+https://github.com/yagoromano/llm-uno.git"
```

### Editable/local development
Clone the repo (or re-clone a fresh copy) and, from the **llm-uno** folder (where `setup.py` lives), run:
```bash
# Clone and enter project
git clone https://github.com/yagoromano/llm-uno.git


cd llm-uno
# Base editable install (for development and single-node examples)
pip install -e .
# Optional: add distributed support (DeepSpeed) for multi-node example
pip install -e .[distributed]
```

### 4. Hugging Face CLI
```bash
pip install huggingface-cli
huggingface-cli login
```

### 5. PyTorch (with CUDA support)


## Repository Structure

```text
llm_uno/                   # Python package with custom game, agents, and OpenRouter
  ├── custom_uno_game.py
  ├── random_agent.py
  └── examples/
      ├── llm_uno_sample.py
      └── llm_dist_sample.py
  └── llm_dist/             # Distributed LLM agent implementation
      └── dist_ClozeAgent.py

llm_uno_sample.py          # Single-node UNO + Hugging Face LLM example
llm_dist_sample.py         # Multi-node distributed UNO + LLM example
README.md                  # This file
ds_config.json             # DeepSpeed config (required)
llama70B_cloze.txt         # Prompt template for Llama-70B
llama8B_cloze.txt          # Prompt template for 8B models
```

## Configuration

Set the Hugging Face cache directory and silence Transformers messages:
```bash
export HF_HOME=/path/to/your/hf_cache
export TRANSFORMERS_VERBOSITY=error
```

## Running the Single-Node Example

After completing the **install** above, you have two ways to run the sample:

1. **Module invocation** (from any folder):
   ```bash
   python3 -m llm_uno.examples.llm_uno_sample
   ```

2. **Script invocation (from editable install)** (from the `examples/` folder):
   ```bash
   cd examples
   python3 llm_uno_sample.py
   ```

This example runs a single UNO game with one LLM player on a single GPU/node. You can use either a Hugging Face model or the OpenRouter agent:

```bash
# Ensure HF_HOME is set
export HF_HOME=/path/to/your/hf_cache

# Run the sample with a Hugging Face LLM
# (Uncomment your chosen model block in llm_uno_sample.py; default models are commented)
python3 llm_uno_sample.py

# Or run with OpenRouter agent (one required argument)
python3 llm_uno_sample.py --api_key YOUR_OPENROUTER_API_KEY
```

- By default the script executes the Hugging Face LLM block. **To switch to a different Hugging Face model or prompting method, uncomment or modify the relevant `model_id` block in `llm_uno_sample.py`. If you switch models, be sure to update the imported prompt template file** (e.g. `llama8B_cloze.txt` or your own template) so your prompt tags align with the new model’s expected format.
- When using OpenRouter, only the `--api_key` flag is required.

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

# single SLURM/torchrun line, placeholder (METHOD)
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --gpus-per-task=$SLURM_GPUS_PER_NODE torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_PER_NODE --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT (Method)


## Replace (METHOD) with one of the following depending on how you installed and where you run from:

# Script invocation (from the llm_uno/examples folder after an editable install):

llm_dist_sample.py

## Module invocation (from any folder, no local clone needed):

-m llm_uno.examples.llm_dist_sample

```

* You can swap in a different script or model by editing `llm_dist_sample.py`.

## Notes

* Ensure that `ds_config.json` (DeepSpeed config) is present.
* Prompt templates and agent parameters should match your chosen model’s token requirements.


## Cite this work
If you find this repo useful, you may cite:
