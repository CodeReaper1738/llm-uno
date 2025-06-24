import rlcard
from rlcard import models
import csv, os
import numpy as np
from rlcard.agents.human_agents.uno_human_agent import HumanAgent, _print_action
from custom_uno_game import CustomUnoGame
from random_agent import RandomAgent
from llm_agent import LLMUnoAgent
from llm_cf_agent import CFLLMUnoAgent

# Configure the environment for 3 players
env = rlcard.make('uno', config={'game_num_players': 2})
env.game.__class__ = CustomUnoGame # Required to correctly calculate payoffs
env.game.configure({'game_num_players': 2})
env.num_players = 2

print(f"Environment configured for {env.num_players} players.")
print(f"Game class: {env.game.__class__.__name__}")

# Load agents
cfr_agent = models.load('uno-rule-v1').agents[0]
random_agent = RandomAgent(env.num_actions)
counterfactual_agent = CFLLMUnoAgent(env.num_actions, model_id="mistralai/Mistral-Small-24B-Instruct-2501")

# Set the agents correctly
#env.set_agents([cfr_agent, cfr_agent, counterfactual_agent])

env.set_agents([cfr_agent, counterfactual_agent])

# Initialize data storage
game_results = []
win_counts = {i: 0 for i in range(env.num_players)}  # Track total wins per player

# Run 100 games
for game_id in range(1, 10001):
    print(f"Starting game {game_id}")

    # Reset the environment for a new game
    env.reset()

    game_duration = 0 
    #played_cards_by_player = {i: [] for i in range(env.num_players)}
    cards_in_hand_history = {i: [] for i in range(env.num_players)}
    played_card_log = []
    game_direction = 1
    llm_all_probs = []
    

    while not env.is_over():
        player_id = env.game.round.current_player
        state = env.game.get_state(player_id)
        
        next_player = (player_id + game_direction) % env.num_players
        llm_next_player = 0 if game_direction == 1 else 1

        # Store number of cards per player
        if 'num_cards' in state:
            for i in range(env.num_players):
                cards_in_hand_history[i].append(state['num_cards'][i])
        else:
            print(f"Warning: 'num_cards' missing from state for Player {player_id}: {state}")

        #if isinstance(env.agents[player_id], LLMUnoAgent):
        if isinstance(env.agents[player_id], (LLMUnoAgent, CFLLMUnoAgent)):
            action, probabilities = env.agents[player_id].step(env._extract_state(state), played_card_log, llm_next_player)
            llm_all_probs.append(probabilities)
        else:
            action = env.agents[player_id].step(env._extract_state(state))
            probabilities = None

        # Convert action to a readable format if necessary
        played_card_log.append((player_id, action))
        
        # Check for reverse card, etc.
        last_played_card = played_card_log[-1][1] if played_card_log else None
        if last_played_card and isinstance(last_played_card, str) and "reverse" in last_played_card:
            game_direction *= -1
            print(f"Reverse played! New turn order: {'Forward' if game_direction == 1 else 'Reversed'}")

        print(f">> Player {player_id} chooses {action}")
        env.game.step(action)
        game_duration += 1


    # Get final results
    payoffs = np.array(env.get_payoffs())

    #print(f"Debug: Game {game_id} Payoffs: {payoffs}")

    # Determine winner(s)
    winner_indices = np.where(payoffs == 1)[0]

    if len(winner_indices) == 0:
        print(f"Warning: No player won in Game {game_id}. Possible timeout or issue in game logic.")
        winner_id = None
        winner = "No winner"
    else:
        winner_id = winner_indices[0]
        winner = f"Player {winner_id}"
        win_counts[winner_id] += 1

    # Assign payoffs correctly
    adjusted_payoffs = np.full(len(payoffs), -1)  # Default -1 for all players
    if winner_id is not None:
        adjusted_payoffs[winner_id] = 1  # Only the winner gets 1

    print(f"Game {game_id} over. Winner: {winner}, Payoffs: {adjusted_payoffs.tolist()}")

    # Store the results
    game_results.append({
        "Game ID": game_id,
        "Winner": winner,
        "Game Duration (Turns)": game_duration, 
        "Player 0 Score (Rand)": adjusted_payoffs[0],
        "Player 1 Score (LLM)": adjusted_payoffs[1],
        "adjusted_payoffs": adjusted_payoffs.tolist(),
        "Cards in Hand History": {i: cards_in_hand_history[i] for i in range(env.num_players)},
        "Played Cards Log": played_card_log,
        "Action Probabilities": llm_all_probs
    })

# Write results to a CSV file
base_filename = "mistral24B_cf"
file_extension = ".csv"
output_file = f"{base_filename}{file_extension}"
file_index = 1

while os.path.exists(output_file):
    output_file = f"{base_filename}_{file_index}{file_extension}"
    file_index += 1

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write header row
    header_row = ["Game ID", "Winner", "Game Duration (Turns)", "Player 0", "Player 1", "Cards in Hand History", "Played Cards Log", "Action Probabilities"]
    writer.writerow(header_row)

    # Write wins row (directly below player headers)
    wins_row = ["Number of Wins", "", ""] + [win_counts[i] for i in range(env.num_players)]
    writer.writerow(wins_row)

    # Write payoffs row (directly below number of wins)
    payoffs_header = ["Final Payoffs", "", "", ""] + ["Payoff"] * env.num_players
    writer.writerow(payoffs_header)

    for result in game_results:
        writer.writerow([
            result["Game ID"],
            result["Winner"],
            result["Game Duration (Turns)"],
            *result["adjusted_payoffs"],
            str(result["Cards in Hand History"]),
            str(result["Played Cards Log"]),   
            str(result["Action Probabilities"]) if result["Action Probabilities"] else ""
        ])

print(f"Results saved to {output_file}")

