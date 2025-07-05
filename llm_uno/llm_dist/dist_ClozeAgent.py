from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import torch.nn.functional as F
import gc
import deepspeed
import torch.distributed as dist
import json
import time
import copy

class DistClozeLLMAgent:
    def __init__(self, num_actions, model_id="meta-llama/Llama-3.3-70B-Instruct", template_path="llama70B_cloze.txt"):
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
        ds_config_path = "ds_config.json"
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

        raw_obs = state['raw_obs']
        legal_actions = raw_obs['legal_actions']
        n = len(legal_actions)
        cumulative_scores = {action: 0.0 for action in legal_actions}

        for shift in range(n):
            shifted = legal_actions[shift:] + legal_actions[:shift]
            state_copy = copy.deepcopy(state)
            state_copy['raw_obs']['legal_actions'] = shifted

            # Tokenize only on main, then broadcast
            if self.is_main_process:
                prompt = self._generate_prompt(state_copy, played_card_log, next_player)
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                obj = [inputs]
            else:
                obj = [None]
            dist.broadcast_object_list(obj, src=0)
            inputs = obj[0]
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            # Memory cleanup
            gc.collect(); torch.cuda.empty_cache()

            if self.is_main_process:
                # Build prefix_allowed_tokens_fn
                legal_token_ids = [self.tokenizer.encode(" " + chr(65 + i), add_special_tokens=False)[0]
                                    for i in range(n)]
                obj = [legal_token_ids]
            else:
                obj = [None]
            dist.broadcast_object_list(obj, src=0)
            legal_token_ids = obj[0]

            # Generation
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    do_sample=True,
                    # prefix_allowed_tokens_fn=lambda batch_id, input_ids: legal_token_ids,
                    temperature=0.4,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            dist.barrier()


            if self.is_main_process:
                # Raw sequence debug
                generated_sequences = outputs.sequences
                input_length = input_ids.shape[1]
                new_tokens = generated_sequences[0][input_length:]
                print("\n=== RAW GENERATED SEQUENCES ===")
                print(f"New tokens decoded: '{self.tokenizer.decode(new_tokens)}'")

                # Compute full vocabulary probabilities
                logits = outputs.scores[0]
                probs = F.softmax(logits, dim=-1)

                # Top-5 overall
                topk = torch.topk(probs, k=5, dim=-1)
                print("\nTop 5 tokens overall:")
                for token_id, prob in zip(topk.indices[0].tolist(), topk.values[0].tolist()):
                    token_str = self.tokenizer.decode([token_id]).strip()
                    print(f"Token ID: {token_id} ('{token_str}'), probability: {prob:.4f}")

                # Legal action letter probabilities
                legal_letters = [chr(65 + i) for i in range(n)]
                legal_letter_token_ids = {}
                for letter in legal_letters:
                    ids = self.tokenizer.encode(" " + letter, add_special_tokens=False)
                    legal_letter_token_ids[letter] = ids[0] if ids else None

                legal_letter_probs = {}
                print("Legal action letter probabilities:")
                for letter, tid in legal_letter_token_ids.items():
                    p = probs[0, tid].item() if (tid is not None and tid < probs.shape[-1]) else 0.0
                    legal_letter_probs[letter] = p
                    print(f"Letter {letter}: {p}")

                # Accumulate shifted-action scores
                for idx, action in enumerate(shifted):
                    letter = chr(65 + idx)
                    cumulative_scores[action] += legal_letter_probs.get(letter, 0.0)

            # Non-main processes skip debug and accumulation

        # Select best action across all shifts
        if self.is_main_process:
            best = max(cumulative_scores, key=cumulative_scores.get)
            print(f"Cumulative scores: {cumulative_scores}")
            print(f"Selected action: {best}")
        else:
            best = None
        action_obj = [best]
        dist.broadcast_object_list(action_obj, src=0)
        final_action = action_obj[0]

        llm_hand = state['raw_obs']['hand']

        return final_action, cumulative_scores, llm_hand
            
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

    def _generate_prompt(self, state, played_card_log, next_player):
        ''' Convert the state into a textual prompt for LLM '''
        # Only generate prompt on main process to save compute
        if not self.is_main_process:
            return ""
            
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
        print(f"Num cards: {num_cards_str}")

        if os.path.exists(self.template_path):
            with open(self.template_path, "r") as f:
                template = f.read()
        else:
            template = """
            <|system|> 
            You are an AI playing UNO. Your goal is to win the game. You are Player {player_id}, and you must make decisions that increase your own chances of winning.

            ### Rules ###
            1. **Action Cards**  
            - Skip: Next player loses a turn  
            - Reverse: Reverses turn order  
            - Draw 2: Next player draws 2 cards and loses a turn  
            - Wild: Choose any color  
            - Wild Draw 4: Choose color + next player draws 4 and loses a turn  
            2. To win the game you must discard all your cards before the other players.

            ### Current Game State ###
            - **Number of Players**: 2 (Player 0, Player 1)
            - **Last Played Card**: {last_played} (played by Player {last_played_by})  
            - **Your Hand**: {hand}  
            - **Next Player**: Player {next_player}  
            - **Recent Moves** (last 5 cards played): {recent_cards}  
            - **Legal Actions**: {legal_actions}

            **Respond ONLY with the letter corresponding to your chosen action from the available legal actions.**

            Your Choice is:
            <|assistant|>"""

        prompt = template.format(
            help_player=1,  # Player to help
            player_id=state['raw_obs']['current_player'],
            last_played=last_played,
            last_played_by=last_played_by,
            hand=", ".join(readable_hand),
            next_player=next_player,
            recent_cards=recent_cards_str,
            legal_actions=action_list_str,
            num_cards=num_cards_str
        )

        print(f"Action list: {action_list_str}")

        print(f"Next player: {next_player}")
        return prompt.strip()