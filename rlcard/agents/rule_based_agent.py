import numpy as np

from rlcard.games.bailongmen.utils import _cards2array, _cards2table, compute_table_score

class RuleBasedAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = True

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        
        
        options = state['raw_legal_actions']
        
        player_id = state['raw_obs']['player_id']
        cur_hand = state['raw_obs']['cur_hand']
        cur_turn = state['raw_obs']['cur_turn']
        board_public = state['raw_obs']['board_public']
        cur_public = state['raw_obs']['cur_public']
        cur_private = state['raw_obs']['cur_private']
        oppo_public = state['raw_obs']['oppo_public']
        oppo_private_count = state['raw_obs']['oppo_private_count']
        
        all_public = board_public + cur_public + oppo_public
        cur_mine = all_public + cur_private
        
        # last turn greedy
        if cur_turn == 11:
            if len(cur_private) < 3: # last slot is private
                try_score = []
                for card in cur_hand:
                    try_cards = cur_mine + card
                    try_score.append(compute_table_score(*_cards2table(try_cards)))
                return cur_hand[np.argmax(try_score)] + '0'
            
            else: # last slot is public
                try_score_gain = []
                for card in cur_hand:
                    try_mine = cur_mine + card
                    try_oppo = all_public + card
                    try_score_gain.append(compute_table_score(*_cards2table(try_mine)) - compute_table_score(*_cards2table(try_oppo)))
                return cur_hand[np.argmax(try_score_gain)] + '1'
        
        # Joker rules
        if 'DW' in all_public and 'XW0' in options:
            return 'XW0'
        if 'XW' in all_public and 'DW0' in options:
            return 'DW0'
        if 'DW' in cur_private:
            if 'XW1' in options:
                return 'XW1'
            if 'XW0' in options:
                return 'XW0'
        if 'XW' in cur_private:
            if 'DW1' in options:
                return 'DW1' 
            if 'DW0' in options:
                return 'DW0'
        if 'DW' in cur_hand and 'XW' in cur_hand:
            if len(cur_private) < 3 and len(cur_public) < 3:
                return 'DW0'
            if len(cur_private) < 2:
                return 'DW0'
            
        # suit rules
        
        # color rules
        
        # first-turn greedy
        
        # general greedy        
            
        
        import IPython; IPython.embed(); exit()
        return np.random.choice(list(state['legal_actions'].keys()))

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
#         probs = [0 for _ in range(self.num_actions)]
#         for i in state['legal_actions']:
#             probs[i] = 1/len(state['legal_actions'])

#         info = {}
#         info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}

        return self.step(state), None # info
