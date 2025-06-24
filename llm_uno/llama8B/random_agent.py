import random

class RandomAgent(object):
    def __init__(self, num_actions):
        self.use_raw = True
        self.num_actions = num_actions

    def step(self, state):
        '''Randomly selects a legal action from the state'''
        # Extract legal actions from the state
        if 'raw_obs' in state:
            # When use_raw=True, legal actions are in state['raw_obs']['legal_actions']
            legal_actions = state['raw_obs']['legal_actions']
        else:
            # When use_raw=False, legal actions are in state['legal_actions']
            legal_actions = list(state['legal_actions'].keys())
        
        # Randomly select and return one legal action
        return random.choice(legal_actions)

    def eval_step(self, state):
        '''Evaluation mode: identical to step'''
        return self.step(state), {}