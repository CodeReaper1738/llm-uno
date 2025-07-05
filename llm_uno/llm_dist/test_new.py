import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch.distributed as dist

def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank

def main():
    rank, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0
    
    # Load tokenizer only on main process
    tokenizer = None
    if is_main:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded")

    # Initialize model with DeepSpeed
    model = deepspeed.init_inference(
        model=AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.3-70B-Instruct",
            torch_dtype=torch.bfloat16
        ),
        config="ds_config.json"
    ).module

    # Define and prepare prompt
    if is_main:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Which programming language do you think is better for AI development?"}
        ]
        
        # Properly format with tokenizer's apply_chat_template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize with attention mask
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        print(f"\nPrompt:\n{formatted_prompt}")
    else:
        inputs = {
            "input_ids": torch.zeros((1, 1), dtype=torch.long).to(device),
            "attention_mask": torch.zeros((1, 1), dtype=torch.long).to(device)
        }

    # Broadcast input length
    seq_len = inputs["input_ids"].shape[1] if is_main else 0
    len_tensor = torch.tensor([seq_len], device=device)
    dist.broadcast(len_tensor, src=0)
    seq_len = len_tensor.item()

    # Prepare tensors on non-main processes
    if not is_main:
        inputs = {
            "input_ids": torch.zeros((1, seq_len), dtype=torch.long).to(device),
            "attention_mask": torch.zeros((1, seq_len), dtype=torch.long).to(device)
        }

    # Broadcast actual tensors
    dist.broadcast(inputs["input_ids"], src=0)
    dist.broadcast(inputs["attention_mask"], src=0)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id if tokenizer else 0
        )

    if is_main:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nResponse:\n{response}\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()