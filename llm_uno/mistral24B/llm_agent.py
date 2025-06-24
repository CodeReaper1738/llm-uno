from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os
import torch.nn.functional as F

class LLMUnoAgent:
    def __init__(self, num_actions, model_id="meta-llama/Llama-3.1-8B"):
        self.use_raw = True
        self.num_actions = num_actions
        self.game_direction = 1

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
        prompt = self._generate_prompt(state, played_card_log, next_player)
        
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

        # --- Compute probabilities for legal action letters only ---
        legal_action_letters = [chr(65 + i) for i in range(len(state['raw_obs']['legal_actions']))]
        legal_token_ids = {}
        for letter in legal_action_letters:
            token_ids = self.tokenizer.encode(letter, add_special_tokens=False) # This is for mistral
            # token_ids = self.tokenizer.encode(" " + letter, add_special_tokens=False) # This is for llama-3.1
            if token_ids:
                legal_token_ids[letter] = token_ids[0]
                #print(f"Token ID for '{letter}': {legal_token_ids[letter]}")
            else:
                legal_token_ids[letter] = None  # Fallback if encoding fails

        legal_letter_probs = {}
        for letter, tid in legal_token_ids.items():
            if tid is not None and tid < probs.shape[-1]:
                legal_letter_probs[letter] = probs[0, tid].item()
            else:
                legal_letter_probs[letter] = 0.0

        print("Legal action letter probabilities:")
        for letter, prob in legal_letter_probs.items():
            print(f"Letter {letter}: {prob}")

        # --- Instead of parsing generated text, choose the highest probability action ---
        best_letter = max(legal_letter_probs, key=legal_letter_probs.get)
        best_index = ord(best_letter) - 65  # 'A' -> index 0, 'B' -> index 1, etc.
        action = state['raw_obs']['legal_actions'][best_index]

        print(f"Selected highest probability action: {action} (letter {best_letter} with probability {legal_letter_probs[best_letter]})")

        return action, legal_letter_probs


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

        template_path = "llm_template.txt"
        if os.path.exists(template_path):
            with open(template_path, "r") as f:
                template = f.read()
        else:
            template = """System: You are an AI playing UNO. Your goal is to help Player {help_player} win the game. You are Player {player_id}, and you must make decisions that increase Player {help_player}'s chances of winning.

            ### Current Game State ###
            - **Last Played Card**: {last_played} (played by Player {last_played_by})
            - **Your Hand**: {hand}
            - **Next Player**: Player {next_player}
            - **Recent Moves** (last 5 cards played):
            {recent_cards}
            - **Legal Actions**: {legal_actions}

            **Respond ONLY with the letter corresponding to your chosen action.** """

        prompt = template.format(
            help_player=0,  # Player to help
            player_id=state['raw_obs']['current_player'],
            last_played=last_played,
            last_played_by=last_played_by,
            hand=", ".join(readable_hand),
            next_player=next_player,
            recent_cards=recent_cards_str,
            legal_actions=action_list_str,
            num_cards=num_cards_str
        )

        print(f"Next player: {next_player}")

        return prompt.strip()