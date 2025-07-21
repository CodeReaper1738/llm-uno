import rlcard
from rlcard import models
from rlcard.agents.human_agents.uno_human_agent import HumanAgent, _print_action

from llm_uno.custom_uno_game import CustomUnoGame
from llm_uno.random_agent import RandomAgent
from llm_uno.llm_dist.dist_ClozeAgent import DistClozeLLMAgent
from llm_uno.llm_dist.dist_CfAgent import DistCFLLMAgent

import torch
import deepspeed
import torch.distributed as dist
import os


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


# Initialize distributed environment
is_main_process, rank, world_size, local_rank = setup_distributed()

env = None
if is_main_process:
    print(f"Distributed setup complete with {world_size} processes")
    print(f"Current process: rank={rank}, local_rank={local_rank}, is_main={is_main_process}")

    # Configure the environment for 3 players
    env = rlcard.make('uno', config={'game_num_players': 3})
    env.game.__class__ = CustomUnoGame
    env.game.configure({'game_num_players': 3})
    env.num_players = 3

    print(f"Environment configured for {env.num_players} players.")
    print(f"Game class: {env.game.__class__.__name__}")
    print(f"Using {world_size} processes for distributed inference.")

# Synchronize all processes
dist.barrier()

if is_main_process:
    rule_agent = models.load('uno-rule-v1').agents[0]
    random_agent = RandomAgent(env.num_actions)
    human_agent = HumanAgent(env.num_actions)
    num_actions = env.num_actions
else:
    rule_agent = None
    random_agent = None
    # All processes need to know num_actions - broadcast from main
    num_actions = 0

# Broadcast num_actions to all processes
num_actions_tensor = torch.tensor([num_actions], dtype=torch.long).cuda()
dist.broadcast(num_actions_tensor, src=0)
num_actions = num_actions_tensor.item()


llm_agent = DistClozeLLMAgent(num_actions, model_id="meta-llama/Llama-3.3-70B-Instruct", template_path="llama70B_cloze.txt", ds_config_path="ds_config.json")
# llm_agent = DistCFLLMAgent(num_actions, model_id="meta-llama/Llama-3.3-70B-Instruct", template_path="llama70B_cf.txt", ds_config_path="ds_config.json")

# Assign agents to players
if is_main_process:
    env.set_agents([rule_agent, rule_agent, llm_agent])

# Synchronize after loading agents
torch.cuda.empty_cache()
dist.barrier()


if is_main_process:

    play_again = True

    while play_again:
        print(">> Start a new game")

        # Reset the environment to initialize the game
        env.reset()
        game_direction = 1
        played_card_log = []

        # Run the game manually to display each action
        while not env.is_over():

            game_over = env.is_over()
            # Get current player ID
            player_id = env.game.round.current_player
            # Get current state for the player
            state = env.game.get_state(player_id)
            
            next_player = (player_id + game_direction) % env.num_players
            llm_next_player = 0 if game_direction == 1 else 1

            # Signal whether this is an LLM turn (-1 = terminate, 0 = non-LLM, 1 = LLM)
            signal = torch.tensor([1 if isinstance(env.agents[player_id], (DistClozeLLMAgent, DistCFLLMAgent)) else 0]).cuda()
            dist.broadcast(signal, src=0)

            if signal.item() == 1:  # LLM turn
                # Broadcast state data for inference
                dist.broadcast_object_list([
                    env._extract_state(state),
                    played_card_log,
                    llm_next_player
                ], src=0)

                # Get action from distributed inference
                action, probabilities, hand = env.agents[player_id].step(
                    env._extract_state(state), 
                    played_card_log, 
                    llm_next_player
                )
            else: 
                action = env.agents[player_id].step(env._extract_state(state))

            played_card_log.append((player_id, action))

            if isinstance(action, str) and "reverse" in action:
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

        # Signal workers to terminate
    dist.broadcast(torch.tensor([-1], device='cuda'), src=0)

else: 
    while True:
        # Wait for the main process to signal a new game
        signal = torch.tensor([0]).cuda()
        dist.broadcast(signal, src=0)
        if signal.item() == -1:
            break
        elif signal.item() == 1: # LLM inference requested
            # Receive inference inputs
            inference_data = [None, None, None]
            dist.broadcast_object_list(inference_data, src=0)
            state, played_card_log, next_player = inference_data

            # Participate in distributed inference
            llm_agent.step(state, played_card_log, next_player)

dist.destroy_process_group()

