import rlcard
from rlcard import models
import csv, os
import numpy as np
from rlcard.agents.human_agents.uno_human_agent import HumanAgent, _print_action
from custom_uno_game import CustomUnoGame
from random_agent import RandomAgent
from dscf_agent import LLMUnoAgent
import torch
import deepspeed
import torch.distributed as dist

def setup_distributed():
    """Initialize the distributed environment."""
    # Read environment variables for distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "4"))  # Total number of processes across all nodes
    
    # Set device
    torch.cuda.set_device(local_rank)

    # Initialize process group if not already
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )   
    
    is_main_process = rank == 0
    
    # Print distributed setup info
    if is_main_process:
        print(f"Distributed setup: MASTER_ADDR={os.environ.get('MASTER_ADDR', 'Not set')}")
        print(f"Distributed setup: MASTER_PORT={os.environ.get('MASTER_PORT', 'Not set')}")
        print(f"Distributed setup: WORLD_SIZE={world_size}, RANK={rank}, LOCAL_RANK={local_rank}")
    
    return is_main_process, rank, world_size, local_rank

if __name__ == "__main__": 
    # Initialize distributed environment
    is_main_process, rank, world_size, local_rank = setup_distributed()
    
    env = None
    if is_main_process:
        print(f"Distributed setup complete with {world_size} processes")
        print(f"Current process: rank={rank}, local_rank={local_rank}, is_main={is_main_process}")
    
        # Configure the environment for 3 players
        env = rlcard.make('uno', config={'game_num_players': 2})
        env.game.__class__ = CustomUnoGame
        env.game.configure({'game_num_players': 2})
        env.num_players = 2

        print(f"Environment configured for {env.num_players} players.")
        print(f"Game class: {env.game.__class__.__name__}")
        print(f"Using {world_size} processes for distributed inference.")

    # Synchronize all processes
    dist.barrier()

    if is_main_process:
        cfr_agent = models.load('uno-rule-v1').agents[0]
        random_agent = RandomAgent(env.num_actions)
        num_actions = env.num_actions
    else:
        cfr_agent = None
        random_agent = None
        # All processes need to know num_actions - broadcast from main
        num_actions = 0
    
    # Broadcast num_actions to all processes
    num_actions_tensor = torch.tensor([num_actions], dtype=torch.long).cuda()
    dist.broadcast(num_actions_tensor, src=0)
    
    # Initialize the LLM agent with proper configuration - ALL processes do this
    llm_agent = LLMUnoAgent(num_actions, model_id="meta-llama/Llama-3.3-70B-Instruct")

    if is_main_process:
        #env.set_agents([cfr_agent, cfr_agent, llm_agent])
        env.set_agents([random_agent, llm_agent])

    # Synchronize after loading agents
    torch.cuda.empty_cache()
    dist.barrier()
    
    # Initialize data storage - only on main process
    if is_main_process:
        game_results = []
        win_counts = {i: 0 for i in range(env.num_players)}
    
    # Run games

    num_games = 10
    if is_main_process:

        for game_id in range(1, num_games + 1):
            
            print(f"Starting game {game_id} of {num_games}")

            # Reset environment
            env.reset()
            game_duration = 0
            cards_in_hand_history = {i: [] for i in range(env.num_players)}
            played_card_log = []
            game_direction = 1
            llm_all_probs = []
            

            while not env.is_over():
                game_over = env.is_over()
                
                player_id = env.game.round.current_player
                state = env.game.get_state(player_id)
                
                next_player = (player_id + game_direction) % env.num_players
                llm_next_player = 0 if game_direction == 1 else 1
        
                if 'num_cards' in state:
                    for i in range(env.num_players):
                        cards_in_hand_history[i].append(state['num_cards'][i])

                # Signal whether this is an LLM turn (-1 = terminate, 0 = non-LLM, 1 = LLM)
                signal = torch.tensor([1 if isinstance(env.agents[player_id], LLMUnoAgent) else 0]).cuda()
                dist.broadcast(signal, src=0)


                if signal.item() == 1:  # LLM turn
                    # Broadcast state data for inference
                    dist.broadcast_object_list([
                        env._extract_state(state),
                        played_card_log,
                        llm_next_player
                    ], src=0)

                    # Get action from distributed inference
                    action, probabilities = llm_agent.step(
                        env._extract_state(state), 
                        played_card_log, 
                        llm_next_player
                    )
                    llm_all_probs.append(probabilities)
                else:  # Non-LLM agent turn
                    action = env.agents[player_id].step(env._extract_state(state))

                played_card_log.append((player_id, action))
                
                if isinstance(action, str) and "reverse" in action:
                    game_direction *= -1
                    print(f"Reverse played! New turn order: {'Forward' if game_direction == 1 else 'Reversed'}")

                    
                print(f">> Player {player_id} chooses {action}")
                
                env.game.step(action)
                game_duration += 1
        
            payoffs = np.array(env.get_payoffs())
            winner_indices = np.where(payoffs == 1)[0]
            winner_id = winner_indices[0] if len(winner_indices) > 0 else None
            adjusted_payoffs = np.full(len(payoffs), -1)
            
            if winner_id is not None:
                adjusted_payoffs[winner_id] = 1
                win_counts[winner_id] += 1
        
            print(f"Game {game_id} over. Winner: {f'Player {winner_id}' if winner_id else 'No winner'}, Payoffs: {adjusted_payoffs.tolist()}")
        
            game_results.append({
                "Game ID": game_id,
                "Winner": f"Player {winner_id}" if winner_id else "No winner",
                "Game Duration (Turns)": game_duration,
                "adjusted_payoffs": adjusted_payoffs.tolist(),
                "Cards in Hand History": {i: cards_in_hand_history[i] for i in range(env.num_players)},
                "Played Cards Log": played_card_log,
                "Action Probabilities": llm_all_probs
            })
        # Signal workers to terminate
        dist.broadcast(torch.tensor([-1], device='cuda'), src=0)
    else:
        # Worker processes - pure inference loop
        while True:
            signal = torch.tensor([0]).cuda()
            dist.broadcast(signal, src=0)
            
            if signal.item() == -1:  # Termination signal
                break
            elif signal.item() == 1:  # LLM inference requested
                # Receive inference inputs
                inference_data = [None, None, None]
                dist.broadcast_object_list(inference_data, src=0)
                state, played_card_log, next_player = inference_data

                # Participate in distributed inference
                _, _ = llm_agent.step(state, played_card_log, next_player)
    
    # Only main process writes results
    if is_main_process:
        base_filename = "dscf_results"
        file_extension = ".csv"
        output_file = f"{base_filename}{file_extension}"
        file_index = 1
    
        while os.path.exists(output_file):
            output_file = f"{base_filename}_{file_index}{file_extension}"
            file_index += 1
    
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header_row = ["Game ID", "Winner", "Game Duration (Turns)", "Player 0", "Player 1", "Cards in Hand History", "Played Cards Log", "Action Probabilities"]
            writer.writerow(header_row)
            
            wins_row = ["Number of Wins", "", ""] + [win_counts[i] for i in range(env.num_players)]
            writer.writerow(wins_row)
            
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

    # Clean up distributed environment
    # Ensure all processes reach this point before cleanup
    dist.barrier()
    dist.destroy_process_group()