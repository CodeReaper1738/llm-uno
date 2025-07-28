from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import torch.nn.functional as F
import gc
import deepspeed
import torch.distributed as dist
import json
import time

class DistCFLLMAgent:
    def __init__(self, num_actions, model_id="meta-llama/Llama-3.3-70B-Instruct", template_path="llama70B_cf.txt", ds_config_path="ds_config.json"):
        self.use_raw = True
        self.num_actions = num_actions
        self.game_direction = 1
        self.template_path = template_path
        
        # Get distributed info from environment variables
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "4"))  # Ensure this matches total GPUs
        
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.is_main_process = self.global_rank == 0

        # Wait for all processes 
        if dist.is_initialized():
            dist.barrier()
        
        # Tokenizer setup (only on main process to save memory)
        start_time = time.time()
        if self.is_main_process:
            print(f"Loading tokenizer for {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
        else: 
            self.tokenizer = None
        
        # Broadcast tokenizer from main process to all others
        if dist.is_initialized():
            dist.barrier()
        
        if self.is_main_process:
            print(f"Beginning model load for {model_id}")
        
        # Memory optimization: clean up before loading model
        gc.collect()
        torch.cuda.empty_cache()

        # --- before deepspeed.init_inference() ---
        if not os.path.exists(ds_config_path):
            raise FileNotFoundError(f"{ds_config_path} not found")

        if self.is_main_process:
            print(f"Loading DeepSpeed config from {ds_config_path}")
        with open(ds_config_path) as f:
            ds_config = json.load(f)

        if self.is_main_process:
            print(f"DeepSpeed config:\n{json.dumps(ds_config, indent=2)}")

        # Ensure all processes have the same model configuration
        if dist.is_initialized():
            dist.barrier()
            
        # Load model 
        start_time = time.time()
            
        # Load model in deepspeed directly
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        # Fix for missing attribute in some models
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for layer in self.model.model.layers:
                if hasattr(layer, "self_attn") and not hasattr(layer.self_attn, "num_heads"):
                    if hasattr(self.model.config, "num_attention_heads"):
                        layer.self_attn.num_heads = self.model.config.num_attention_heads
        
        # Initialize DeepSpeed Inference engine
        self.model = deepspeed.init_inference(
            model=self.model,
            config=ds_config
        )
        self.model.eval()

        if dist.is_initialized():
            dist.barrier()

        # print per-rank allocated GPU memory
        alloc_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"[rank {self.global_rank}/{self.world_size}] "
            f"local_rank={self.local_rank} → {alloc_gb:.2f} GB on GPU {self.local_rank}")
        
        if self.is_main_process:
            print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        
        # Memory optimization: clean up after initialization
        gc.collect()
        torch.cuda.empty_cache()
        
        # Wait for all processes to finish loading
        if dist.is_initialized():
            dist.barrier()
            
        if self.is_main_process:
            print("Model fully initialized and ready")

    def step(self, state, played_card_log, next_player):
        # Generate prompt and tokenize ONLY on main process

        legal = state['raw_obs']['legal_actions']
        n = len(legal)
        scores = {action: 0.0 for action in legal}

        for candidate in legal:
            # build prompt
            if self.is_main_process:
                prompt = self._generate_prompt(state, played_card_log, next_player, candidate)
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                obj = [inputs]
            else:
                obj = [None]
            dist.broadcast_object_list(obj, src=0)
            inputs = obj[0]
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            # inference
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=0.6,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            dist.barrier()


            if self.is_main_process:
                generated_sequences = outputs.sequences
                # Get length of input prompt
                input_length = inputs.input_ids.shape[1]
                # Extract only the newly generated tokens
                new_tokens = generated_sequences[0][input_length:]
                # print("\n=== RAW GENERATED SEQUENCES ===")
                # print(f"\nNew tokens only (IDs): {new_tokens.tolist()}")
                # print(f"New tokens decoded: '{self.tokenizer.decode(new_tokens, skip_special_tokens=True)}'")

                # Get logits and softmax
                logits = outputs.scores[0]
                probs = F.softmax(logits, dim=-1)

                # Top-5 tokens
                topk = torch.topk(probs, k=5, dim=-1)
                # print(f"\nTop 5 tokens for candidate {candidate}:")
                for token_id, prob in zip(topk.indices[0].tolist(), topk.values[0].tolist()):
                    token_str = self.tokenizer.decode([token_id]).strip()
                    # print(f"Token ID: {token_id} ('{token_str}'), probability: {prob:.4f}")

                good_variant_ids = []
                for v in (" good", "good"):
                    toks = self.tokenizer.encode(v, add_special_tokens=False)
                    if len(toks) == 1:
                        good_variant_ids.append(toks[0])

                bad_variant_ids = []
                for v in (" bad", "bad"):
                    toks = self.tokenizer.encode(v, add_special_tokens=False)
                    if len(toks) == 1:
                        bad_variant_ids.append(toks[0])

                # pick the highest-prob variant in each set
                good_id = max(good_variant_ids, key=lambda tid: probs[0, tid].item())
                bad_id  = max(bad_variant_ids,  key=lambda tid: probs[0, tid].item())
                good_prob = probs[0, good_id].item()
                bad_prob  = probs[0, bad_id].item()

                score = good_prob - bad_prob
                scores[candidate] = score
                # print(f"Candidate: {candidate}, good:{good_prob:.4f}, bad:{bad_prob:.4f}, score:{score:.4f}")

        if self.is_main_process:
            best_candidate = max(scores, key=scores.get)
            best_action = best_candidate
            print(f"Best action chosen: {best_action} with score {scores[best_action]:.4f}")
            action_obj = [best_action]
        else:
            action_obj = [None] # Placeholder for action object

        dist.broadcast_object_list(action_obj, src=0)
        final_action = action_obj[0]

        llm_hand = state['raw_obs']['hand']

        return final_action, scores, llm_hand
            
    def eval_step(self, state):
        ''' Evaluation mode: identical to `step` here. '''
        return self.step(state), {}

    def convert_card_format(self, card):
        color_map = {
            'r': 'red',
            'g': 'green',
            'b': 'blue',
            'y': 'yellow'
        }

        parts = card.split('-')
        if len(parts) == 2:
            color, trait = parts
            return f"{color_map.get(color, color)} {trait}"
        return card

    def _generate_prompt(self, state, played_card_log, next_player, candidate):
        ''' Convert the state into a textual prompt for LLM '''
        last_played = self.convert_card_format(state['raw_obs']['target'])
        last_played_by = played_card_log[-1][0] if played_card_log else "None" 
        readable_hand = [self.convert_card_format(card) for card in state['raw_obs']['hand']]
        readable_legal_actions = {chr(65 + i): self.convert_card_format(action) for i, action in enumerate(state['raw_obs']['legal_actions'])} 

        print(f"Readable legal actions: {readable_legal_actions}")

        # Format recent moves (last 5 cards played)
        recent_moves = [
            f"- Player {player}: {self.convert_card_format(card)}"
            for player, card in played_card_log[-5:]  # Extract last 5 moves
        ]
        recent_cards_str = "\n  ".join(recent_moves) if recent_moves else "No recent moves"

        # Create a mapping from letters to legal actions
        action_list_str = "\n  ".join([f"{letter}: {action}" for letter, action in readable_legal_actions.items()])

        num_cards_str = "\n  ".join([f"Player {i}: {count}" for i, count in enumerate(state['raw_obs']['num_cards'])])
        #print(f"Num cards: {num_cards_str}")


        if os.path.exists(self.template_path):
            with open(self.template_path, "r") as f:
                template = f.read()
        else:
            template = """
            <s>[INST] <<SYS>>
You are an AI playing UNO. Your goal is to help Player {help_player} win the game. You are Player {player_id}, and you must make decisions that increase Player {help_player}’s chances of winning.
<</SYS>>

### Rules ###
1. **Action Cards** 
   - Skip: Next player loses a turn
   - Reverse: Reverses turn order
   - Draw 2: Next player draws 2 cards and loses a turn
   - Wild: Choose any color
   - Wild Draw 4: Choose color + next player draws 4 and loses a turn
2. To win the game you must discard all your cards before the other players

### Current Game State ###
- **Number of Players**: 3 (Player 0, Player 1, Player 2)
- **Last Played Card**: {last_played} (played by Player {last_played_by})
- **Your Hand**: {hand}
- **Next Player**: Player {next_player}
- **Recent Moves** (last 5 cards played): 
  {recent_cards}
- **Legal Actions**: 
  {legal_actions}

Evaluate answer choice {candidate} as either "good" if it helps Player {help_player} win, otherwise "bad". 
Respond ONLY with either "good" or "bad".

Your answer:
[/INST]</s>"""

        prompt = template.format(
            help_player=1,  # Player to help
            player_id=state['raw_obs']['current_player'],
            last_played=last_played,
            last_played_by=last_played_by,
            hand=", ".join(readable_hand),
            next_player=next_player,
            recent_cards=recent_cards_str,
            legal_actions=action_list_str,
            num_cards=num_cards_str,
            candidate=candidate
        )

        # print(f"Next player: {next_player}")

        return prompt.strip()