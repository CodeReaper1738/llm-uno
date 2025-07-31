import rlcard
from rlcard import models
from rlcard.agents.human_agents.uno_human_agent import HumanAgent, _print_action

import argparse
from llm_uno.custom_uno_game import CustomUnoGame
from llm_uno.random_agent import RandomAgent

from llm_uno.local_ClozeAgent import LocalClozeLLMAgent
from llm_uno.local_CfAgent import LocalCFLLMAgent
from llm_uno.openrouter import OpenRouter


parser = argparse.ArgumentParser(description="Run UNO AI with DeepSeek R1 via OpenRouter.")
parser.add_argument("--api-key", type=str, help="OpenRouter API Key.")
args = parser.parse_args()

api_key = args.api_key

# Make environment with 3 players
env = rlcard.make('uno', config={'game_num_players': 3})
env.game.configure({'game_num_players': 3})
env.num_players = 3

# Initialize agents
human_agent = HumanAgent(env.num_actions)
rule_agent = models.load('uno-rule-v1').agents[0]
random_agent = RandomAgent(env.num_actions)

# Only initialize one LLM agent 
llm_agent = LocalClozeLLMAgent(env.num_actions, model_id="meta-llama/Llama-3.1-8B", template_path="llama_cloze.txt")
# llm_agent = LocalCFLLMAgent(env.num_actions, model_id="meta-llama/Llama-3.1-8B", template_path="llama_cf.txt")


# open_router_agent = OpenRouter(env.num_actions, api_key, model_id="deepseek/deepseek-chat:free", template_path="llama8B_cloze.txt")

# Assign agents to players
env.set_agents([rule_agent, rule_agent, llm_agent])

print(">> UNO rule model V1")
play_again = True

while play_again:
    print(">> Start a new game")

    # Reset the environment to initialize the game
    env.reset()
    
    played_cards_by_player = {i: [] for i in range(env.num_players)}
    game_direction = 1
    played_card_log = []
    # Run the game manually to display each action
    while not env.is_over():
        # Get current player ID
        player_id = env.game.round.current_player

        # Get current state for the player
        state = env.game.get_state(player_id)
        
        next_player = (player_id + game_direction) % env.num_players
        
        llm_next_player = 0 if game_direction == 1 else 1

        # Determine the action for the current player
        if isinstance(env.agents[player_id], (LocalClozeLLMAgent, LocalCFLLMAgent)):
            action, _, _ = env.agents[player_id].step(env._extract_state(state), played_card_log, llm_next_player)
        elif isinstance(env.agents[player_id], OpenRouter):
            action, _ = env.agents[player_id].step(env._extract_state(state), played_card_log, llm_next_player)
        else:
            action = env.agents[player_id].step(env._extract_state(state))

        if action:
            played_card_log.append((player_id, action))
            played_cards_by_player[player_id].append(action)
        else:
            print(f"Warning: Action was not assigned for Player {player_id}, skipping log entry.")
                   
            
        last_played_card = played_card_log[-1][1] if played_card_log else None
        if last_played_card and "reverse" in last_played_card:
            game_direction *= -1 
            print(f"Reverse played! New turn order: {'Forward' if game_direction == 1 else 'Reversed'}")
        
        # Log and print the action
        print(f">> Player {player_id} chooses ", end="")
        _print_action(action)
        print()

        # Step the game forward with the chosen action
        env.game.step(action)

    # Get payoffs after the game ends
    payoffs = env.get_payoffs()

    # Determine the winning player
    winning_player = max(range(len(payoffs)), key=lambda i: payoffs[i])

    # Display only the winning player
    print('===============     Result     ===============')
    print(f"Player {winning_player} wins the game!")

    resp = input("Play again? (y/n): ")
    if resp.lower() not in ("y", "yes"):
        play_again = False

