from openai import OpenAI
import os, time

class OpenRouter:
    def __init__(self, num_actions, api_key, model_id="deepseek/deepseek-chat:free", template_path="llama8B_cloze.txt"):
        self.use_raw = True
        self.num_actions = num_actions
        self.game_direction = 1
        self.template_path = template_path

        self.model_id = model_id
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def step(self, state, played_card_log, next_player):
        ''' Step function where DeepSeek selects an action '''

        raw_obs = state['raw_obs']
        legal_actions = raw_obs['legal_actions']
        # Map letters A, B, C... to actions
        letter_map = {chr(65+i): action for i, action in enumerate(legal_actions)}

        time.sleep(3)
        while True:
            # Build prompt
            prompt = self._generate_prompt(state, played_card_log, next_player)
            #print(prompt)
            # Call text completion for one token
            try:
                response = self.client.completions.create(
                    model=self.model_id,
                    prompt=prompt,
                    max_tokens=1,
                    temperature=0.1,
                    logprobs=0
                )
            except Exception as e:
                print(f"API error: {e}, retrying...")
                time.sleep(0.5)
                continue

            choice = response.choices[0]
            action_text = choice.text.strip()
            letter = action_text.upper()
            print(f"Raw API output: '{action_text}'")

            if letter in letter_map:
                chosen = letter_map[letter]
                print(f"Selected action: {chosen} (letter {letter})")
                break
            else:
                print(f"Invalid response '{action_text}', retrying...")
                time.sleep(0.5)

        # Return the chosen action, empty scores, and current hand
        llm_hand = raw_obs['hand']
        return chosen, llm_hand

    def eval_step(self, state):
        ''' Evaluation mode: identical to `step`. '''
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
