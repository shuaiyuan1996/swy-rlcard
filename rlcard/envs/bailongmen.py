from collections import Counter, OrderedDict
import numpy as np

from rlcard.envs import Env
from rlcard.games.bailongmen.game import SwyBlmGame
from rlcard.games.bailongmen.utils import ACTION_2_ID, ID_2_ACTION
from rlcard.games.bailongmen.utils import _cards2array, _cards2table


def one_hot(i, n):
    array = np.zeros(int(n))
    array[int(i)] = 1.
    return array
        
        
class SwyBlmEnv(Env):
    ''' Swy-blm Environment
    '''

    def __init__(self, config):        
        self.name = 'swy-blm'
        self.game = SwyBlmGame()
        super().__init__(config)
        self.state_shape = [[230], [230]]
        self.action_shape = [[44], [44]]
        
    def _extract_state(self, state):
        ''' Encode state

        Args:
            state (dict): dict of original state
        '''
        
        # encode cards on different sections
        board_public = _cards2array(state['board_public'])
        cur_public = _cards2array(state['cur_public'])
        cur_private = _cards2array(state['cur_private'])
        oppo_public = _cards2array(state['oppo_public'])
        cur_hand = _cards2array(state['cur_hand'])
        
        # encode empty space
        cur_public_space = one_hot(3 - len(state['cur_public']), 4)
        cur_private_space = one_hot(3 - len(state['cur_private']), 4)
        oppo_public_space = one_hot(3 - len(state['oppo_public']), 4)
        oppo_private_space = one_hot(3 - state['oppo_private_count'], 4)
        
        # encode score table
        cur_table, cur_joker = _cards2table(state['board_public'] + state['cur_public'] + state['oppo_public'] + state['cur_private'])
        oppo_table, oppo_joker = _cards2table(state['board_public'] + state['cur_public'] + state['oppo_public'])
       
        cur_joker_count = one_hot(cur_joker.sum(), 3)
        cur_suit_count = np.zeros(4 * 6)
        for i in range(4):  # four different suits
            cur_suit_count[i*6: (i+1)*6] = one_hot(cur_table[:, i].sum(), 6)
        cur_color_count = np.zeros(5 * 5)
        for i in range(5):  # five different colors
            cur_color_count[i*5: (i+1)*5] = one_hot(cur_table[i, :].sum(), 5)
            
        oppo_joker_count = one_hot(oppo_joker.sum(), 3)
        oppo_suit_count = np.zeros(4 * 6)
        for i in range(4):  # four different suits
            oppo_suit_count[i*6: (i+1)*6] = one_hot(oppo_table[:, i].sum(), 6)
        oppo_color_count = np.zeros(5 * 5)
        for i in range(5):  # five different colors
            oppo_color_count[i*5: (i+1)*5] = one_hot(oppo_table[i, :].sum(), 5)
            
        obs = np.concatenate((board_public, cur_public, cur_private, oppo_public, cur_hand,
                              cur_public_space, cur_private_space, oppo_public_space, oppo_private_space,
                              cur_joker_count, cur_suit_count, cur_color_count,
                              oppo_joker_count, oppo_suit_count, oppo_color_count))
        
        # encode legal actions
        raw_legal_actions = self._get_legal_actions()
        legal_actions = {ACTION_2_ID[x]: one_hot(ACTION_2_ID[x], len(ID_2_ACTION)) for x in raw_legal_actions}

        extracted_state = OrderedDict({'obs': obs, 'legal_actions': legal_actions})
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = raw_legal_actions
        extracted_state['action_record'] = self.action_recorder
        return extracted_state
            
    def get_payoffs(self):
        ''' Get the payoffs of players. Must be implemented in the child class.

        Returns:
            payoffs (list): a list of payoffs for each player
        '''
        score_diff = self.game.final_score[0] - self.game.final_score[1]     
        return np.array([score_diff, -score_diff], dtype=np.float32)

    def _decode_action(self, action_id):
        ''' Action id -> the action in the game. Must be implemented in the child class.

        Args:
            action_id (int): the id of the action

        Returns:
            action (string): the action that will be passed to the game engine.
        '''
        return ID_2_ACTION[action_id]

    def _get_legal_actions(self):
        ''' Get all legal actions for current state

        Returns:
            legal_actions (list): a list of legal actions' id
        '''
        state = self.game.current_state
        legal_actions = self.game.compute_legal_actions(state['cur_hand'], len(state['cur_public']), len(state['cur_private']))
        return legal_actions

    def get_action_feature(self, action):
        ''' For some environments such as DouDizhu, we can have action features
        Returns:
            (numpy.array): The action features
        '''
        return one_hot(action, len(ID_2_ACTION))
        