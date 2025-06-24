import torch
import deepspeed
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch.distributed as dist
import matplotlib.pyplot as plt

def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "4"))
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank

def main():
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0
    
    # Only main process handles tokenizer
    tokenizer = None
    if is_main:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded on main process")

    # Initialize model with DeepSpeed
    model = deepspeed.init_inference(
        model=AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.3-70B-Instruct",
            torch_dtype=torch.float16
        ),
        config="ds_config.json"
    ).module
    
    # Generate text with proper formatting
    system_message = "You are a helpful assistant."
    user_message = "Hello, my dog looks cute"
    max_new_tokens = 25
    
    if is_main:
        # Format prompt with Llama-3 special tokens
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
        input_size = input_ids.shape[1]
        print(f"Formatted prompt: {formatted_prompt}")
        print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
    else:
        input_ids = torch.zeros((1, 1), dtype=torch.long).to(device)
        input_size = 0
    
    # Broadcast input size
    size_tensor = torch.tensor([input_size], device=device)
    dist.broadcast(size_tensor, src=0)
    input_size = size_tensor.item()
    
    # Broadcast input IDs
    if not is_main:
        input_ids = torch.zeros((1, input_size), dtype=torch.long).to(device)
    dist.broadcast(input_ids, src=0)
    
    # Generation loop
    generated_text = user_message
    for i in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
        
        next_token_logits = outputs.logits[:, -1, :]
        
        # Sample next token (top-p sampling)
        next_token = sample_top_p(next_token_logits, top_p=0.9)
        
        # Update input for next iteration
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        if is_main:
            if i < 5:  # Visualize first few steps
                visualize_top_tokens(next_token_logits, tokenizer, i + 1)
            new_token = tokenizer.decode(next_token[0])
            generated_text += new_token
            print(f"Step {i+1}: {generated_text}")

    if is_main:
        print("\nFinal output:", generated_text)

def visualize_top_tokens(logits, tokenizer, step_num):
    """Visualize top probable tokens for a given step"""
    if not tokenizer:  # Skip if not main process
        return
    
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()
    
    # Get top 50 tokens
    topk = torch.topk(logits, k=50, dim=-1)
    topk_probs = probs[0, topk.indices[0].cpu().numpy()]
    tokens = tokenizer.convert_ids_to_tokens(topk.indices[0])
    
    # Clean token representations
    clean_tokens = []
    for t in tokens:
        t = t.replace('Ġ', ' ')  # Handle space markers
        t = t.replace('<0x0A>', '\\n')  # Handle newlines
        t = t.strip()
        clean_tokens.append(t)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.bar(clean_tokens, topk_probs)
    plt.xticks(rotation=90)
    plt.ylabel("Probability")
    plt.title(f"Top 50 Token Probabilities - Step {step_num}")
    plt.tight_layout()
    plt.savefig(f"token_probs_step_{step_num}.png")
    plt.close()
    print(f"Saved token probabilities visualization for step {step_num}")

def sample_top_p(logits, top_p=0.9):
    """Top-p sampling (nucleus sampling)"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens outside top-p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, 
        index=sorted_indices, 
        src=sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')
    
    # Sample from remaining tokens
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

if __name__ == "__main__":
    main()
    dist.destroy_process_group()