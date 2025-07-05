from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os
import torch.nn.functional as F
import copy

class LocalClozeLLMAgent:
    def __init__(self, num_actions, model_id="meta-llama/Llama-3.1-8B", template_path="llama8B_cloze.txt"):
        self.use_raw = True
        self.num_actions = num_actions
        self.game_direction = 1

        # Store external template if provided
        self.template_path = template_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Ensure a padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS token as padding
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

        # Print device information
        print(f"Model {model_id} is running on: {next(self.model.parameters()).device}")
        print(f"Model is loaded across these devices: {self.model.hf_device_map}")

    def step(self, state, played_card_log, next_player):
        ''' Step function where LLM selects an action '''

        raw_obs = state['raw_obs']
        legal_actions = raw_obs['legal_actions']
        n = len(legal_actions)


        cumulative_scores = {action: 0.0 for action in legal_actions}


        for shift in range(n):
            # Shift the legal actions to simulate different perspectives
            shifted_actions = legal_actions[shift:] + legal_actions[:shift]
            state_copy = copy.deepcopy(state)
            state_copy['raw_obs']['legal_actions'] = shifted_actions

            # Generate the prompt for the current perspective

            prompt = self._generate_prompt(state_copy, played_card_log, next_player)
        
            # Tokenize the prompt and move inputs to the model's device
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=True,
                temperature=0.4,
                output_scores=True,
                return_dict_in_generate=True
            )

            generated_sequences = outputs.sequences

            # Get the length of the input prompt
            input_length = inputs.input_ids.shape[1]

            # Extract only the newly generated tokens (after the prompt)
            new_tokens = generated_sequences[0][input_length:]

            print("\n=== RAW GENERATED SEQUENCES ===")
            print(f"\nNew tokens only (IDs): {new_tokens.tolist()}")
            print(f"New tokens decoded: '{self.tokenizer.decode(new_tokens, skip_special_tokens=True)}'")


            # Compute probabilities for the first generated token.
            logits = outputs.scores[0]  # shape: [1, vocab_size]
            probs = F.softmax(logits, dim=-1) # shape: [1, vocab_size]

            # Right after computing probs = F.softmax(logits, dim=-1)
            topk = torch.topk(probs, k=5, dim=-1)
            print("\nTop 5 tokens overall:")
            for token_id, prob in zip(topk.indices[0].tolist(), topk.values[0].tolist()):
                token_str = self.tokenizer.decode([token_id]).strip()
                print(f"Token ID: {token_id} ('{token_str}'), probability: {prob:.4f}")

            # d) Compute legal‐letter probs for this shift
            letters = [chr(65 + i) for i in range(len(shifted_actions))]
            letter_token_ids = {}
            for L in letters:
                # collect single-token variants
                variant_ids = []
                for variant in (f" {L}", L):
                    toks = self.tokenizer.encode(variant, add_special_tokens=False)
                    if len(toks) == 1:
                        variant_ids.append(toks[0])
                # pick variant with highest probability
                best_id = max(variant_ids, key=lambda tid: probs[0, tid].item())
                letter_token_ids[L] = best_id

            for idx, action in enumerate(shifted_actions):
                L = letters[idx]
                tid = letter_token_ids[L]
                p = probs[0, tid].item()
                print(f"[Shift {shift}] Letter {L}: {p:.4f} → action '{action}'")
                # e) Accumulate into the action’s total
                cumulative_scores[action] += p

        # 3) After all shifts: pick the action with the highest total score
        print("\n=== CUMULATIVE SCORES ===")
        for act, score in cumulative_scores.items():
            print(f"  {act}: {score:.4f}")
        best = max(cumulative_scores, key=cumulative_scores.get)
        print(f"→ Selected: {best}")

        llm_hand = state['raw_obs']['hand']

        return best, cumulative_scores, llm_hand


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
            template = """You are an AI playing UNO. Your goal is to help Player {help_player} win the game. You are Player {player_id}, and you must make decisions that increase Player {help_player}’s chances of winning.

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
                - **Number of Cards per Player**: {num_cards} 
                - **Last Played Card**: {last_played} (played by Player {last_played_by})  
                - **Your Hand**: {hand}  
                - **Next Player**: Player {next_player}  
                - **Recent Moves** (last 5 cards played): {recent_cards}  
                - **Legal Actions**: {legal_actions}

                **Respond ONLY with the letter corresponding to your chosen action from the available legal actions.**

                Your Choice is:"""

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
        
        # print(f"Next player: {next_player}")

        return prompt.strip()