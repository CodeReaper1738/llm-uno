from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os
import torch.nn.functional as F

class LocalCFLLMAgent:
    def __init__(self, num_actions, model_id="meta-llama/Llama-3.1-8B", template_path="llama8B_cf.txt"):
        self.use_raw = True
        self.num_actions = num_actions
        self.game_direction = 1

        self.template_path = template_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        
        print(f"Model {model_id} running on: {next(self.model.parameters()).device}")
        print(f"Model is loaded across these devices: {self.model.hf_device_map}")

    def step(self, state, played_card_log, next_player):
        
        legal = state['raw_obs']['legal_actions']
        n = len(legal)
        scores = {action: 0.0 for action in legal}


        for shift in range(n):

            shifted_actions = legal[shift:] + legal[:shift]
            candidate = shifted_actions[0]  # The first action in the shifted list is the candidate

            prompt = self._generate_prompt(state, played_card_log, next_player, candidate, shifted_actions)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
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

            # Get logits for the first generated token
            logits = outputs.scores[0]  # shape: [1, vocab_size]
            probs = F.softmax(logits, dim=-1)  # shape: [1, vocab_size]

            topk = torch.topk(probs, k=5, dim=-1) # Get top 5 tokens
            print(f"\nTop 5 tokens for candidate {candidate}:")
            for token_id, prob in zip(topk.indices[0].tolist(), topk.values[0].tolist()):
                token_str = self.tokenizer.decode([token_id]).strip()
                print(f"Token ID: {token_id} ('{token_str}'), probability: {prob:.4f}")

            # Compute probability for "good" and "bad" handling both variants
            good_variant_ids = []
            for variant in (" good", "good"):
                toks = self.tokenizer.encode(variant, add_special_tokens=False)
                if len(toks) == 1:
                    good_variant_ids.append(toks[0])
            bad_variant_ids = []
            for variant in (" bad", "bad"):
                toks = self.tokenizer.encode(variant, add_special_tokens=False)
                if len(toks) == 1:
                    bad_variant_ids.append(toks[0])
            if not good_variant_ids or not bad_variant_ids:
                raise ValueError("Cannot encode 'good' or 'bad' as single token")
            # pick highest-prob variant for each
            good_id = max(good_variant_ids, key=lambda tid: probs[0, tid].item())
            bad_id  = max(bad_variant_ids,  key=lambda tid: probs[0, tid].item())

            good_prob = probs[0, good_id].item()
            bad_prob  = probs[0, bad_id].item()
            score = good_prob - bad_prob
            scores[candidate] = score

            print(f"Candidate: {candidate}, good:{good_prob:.4f}, bad:{bad_prob:.4f}, score:{score:.4f}")

        best_candidate = max(scores, key=scores.get)
        best_action = best_candidate
        print(f"Best action chosen: {best_action} with score {scores[best_action]:.4f}")

        llm_hand = state['raw_obs']['hand']

        return best_action, scores, llm_hand

    def eval_step(self, state):
        # For evaluation, call step with empty played_card_log and default next_player=0.
        return self.step(state, played_card_log=[], next_player=0)

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

    def _generate_prompt(self, state, played_card_log, next_player, candidate, shifted_actions):
        ''' Convert the state into a textual prompt for LLM '''
        last_played = self.convert_card_format(state['raw_obs']['target'])
        last_played_by = played_card_log[-1][0] if played_card_log else "None" 
        readable_hand = [self.convert_card_format(card) for card in state['raw_obs']['hand']]
        readable_legal_actions = [ 
            self.convert_card_format(action) 
            for action in shifted_actions
        ]

        print(f"Readable legal actions: {readable_legal_actions}")

        # Format recent moves (last 5 cards played)
        recent_moves = [
            f"- Player {player}: {self.convert_card_format(card)}"
            for player, card in played_card_log[-5:]  # Extract last 5 moves
        ]
        recent_cards_str = "\n  ".join(recent_moves) if recent_moves else "No recent moves"

        # Create a mapping from letters to legal actions
        action_list_str = "\n  ".join(readable_legal_actions)

        num_cards_str = "\n  ".join([f"Player {i}: {count}" for i, count in enumerate(state['raw_obs']['num_cards'])])
        #print(f"Num cards: {num_cards_str}")

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

Evaluate answer choice {candidate} as either "good" if it helps Player {help_player} win, otherwise "bad". 
Respond ONLY with either "good" or "bad".

Your Choice is: """

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
            candidate=self.convert_card_format(candidate)
        )

        # print(f"Prompt for candidate '{candidate}':\n{prompt}\n")
        print(f"Next player: {next_player}")

        return prompt.strip()
