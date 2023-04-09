import numpy as np

from rlcard.games.bailongmen.utils import SUIT, COLOR
from rlcard.games.bailongmen.utils import _cards2array, _cards2table, compute_table_score
from rlcard.games.bailongmen.utils import count_suit, count_color

class RuleBasedAgent(object):
    ''' A rule-based agent. We implement rules from experts as a heuristic strategy that can be used in validation.
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
        
        ## last turn greedy
        if 10 <= cur_turn <= 11:
            if len(cur_private) < 3: # last slot is private
                try_score = []
                for card in cur_hand:
                    try_cards = cur_mine + [card]
                    try_score.append(compute_table_score(*_cards2table(try_cards)))
                return cur_hand[np.argmax(try_score)] + '0'
            
            else: # last slot is public
                try_score_gain = []
                for card in cur_hand:
                    try_mine = cur_mine + [card]
                    try_oppo = all_public + [card]
                    try_score_gain.append(compute_table_score(*_cards2table(try_mine)) - compute_table_score(*_cards2table(try_oppo)))
                return cur_hand[np.argmax(try_score_gain)] + '1'
                
        ## Joker rules
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
            
        hand_suit_nums = []
        cur_private_suit_nums = []
        all_public_suit_nums = []
        hand_suit_cards = []
        for suit in SUIT:
            hand_suit_nums.append(min(count_suit(cur_hand, suit), 6 - len(cur_public) - len(cur_private)))      
            cur_private_suit_nums.append(count_suit(cur_private, suit))
            all_public_suit_nums.append(count_suit(all_public, suit))
            hand_suit_cards.append(list(filter(lambda x: suit in x, cur_hand)))
            
        hand_color_nums = []
        cur_private_color_nums = []
        all_public_color_nums = []
        hand_color_cards = []
        for color in COLOR:
            hand_color_nums.append(min(count_color(cur_hand, color), 6 - len(cur_public) - len(cur_private)))      
            cur_private_color_nums.append(count_color(cur_private, color))
            all_public_color_nums.append(count_color(all_public, color))
            hand_color_cards.append(list(filter(lambda x: color in x, cur_hand)))
             
        
        ## suit rules
        # one suit card in hand, 4-5 in total: immediate score
        for suit, hand_suit_num, all_public_suit_num, cur_private_suit_num, hand_suit_card in zip(SUIT, hand_suit_nums, all_public_suit_nums, cur_private_suit_nums, hand_suit_cards):
            if hand_suit_num == 1:
                preference = []
                idx = np.argsort((np.array(all_public_color_nums) + np.array(cur_private_color_nums))[3:5])  # C/R
                for i in idx[::-1].tolist():
                    if COLOR[i+3] + suit in hand_suit_card:
                        preference.append(COLOR[i+3] + suit) 
                idx = np.argsort((np.array(all_public_color_nums) + np.array(cur_private_color_nums))[:3])  # G/B/Y
                for i in idx[::-1].tolist():
                    if COLOR[i] + suit in hand_suit_card:
                        preference.append(COLOR[i] + suit)              
                preferred_card = preference[0]
                nonpreferred_card = preference[-1]
                
                if all_public_suit_num + cur_private_suit_num + hand_suit_num == 5: # 5 same suit: prefer 2 private, 3 public
                    if 3 <= all_public_suit_num <= 4:
                        if len(cur_private) < 3:  # need to put in private; otherwise, not necessary
                            return preferred_card + '0'
                        else:
                            pass
                    elif 1<= all_public_suit_num <= 2:
                        if len(cur_public) < 3:  # prefer to put in public
                            return nonpreferred_card + '1'
                        else:
                            return preferred_card + '0'
                    else:
                        pass # impossible
                    
                if all_public_suit_num + cur_private_suit_num + hand_suit_num == 4: # 4 same suit: prefer 2 private, 2 public
                    if 2 <= all_public_suit_num <= 3:
                        if len(cur_private) < 3:  # need to put in private; otherwise, not necessary
                            return preferred_card + '0'
                        else:
                            pass
                    else:
                        if len(cur_public) < 3:  # prefer to put in public
                            return nonpreferred_card + '1'
                        else:
                            return preferred_card + '0'                        

        # two suit cards in hand, 4-5 in total: close to immediate score
        for suit, hand_suit_num, all_public_suit_num, cur_private_suit_num, hand_suit_card in zip(SUIT, hand_suit_nums, all_public_suit_nums, cur_private_suit_nums, hand_suit_cards):
            if hand_suit_num == 2:
                preference = []
                idx = np.argsort((np.array(all_public_color_nums) + np.array(cur_private_color_nums))[3:5])  # C/R
                for i in idx[::-1].tolist():
                    if COLOR[i+3] + suit in hand_suit_card:
                        preference.append(COLOR[i+3] + suit) 
                idx = np.argsort((np.array(all_public_color_nums) + np.array(cur_private_color_nums))[:3])  # G/B/Y
                for i in idx[::-1].tolist():
                    if COLOR[i] + suit in hand_suit_card:
                        preference.append(COLOR[i] + suit)              
                preferred_card = preference[0]
                nonpreferred_card = preference[-1]
                
                if 'C' + suit in hand_suit_card:
                    preferred_card = 'C' + suit
                elif 'R' + suit in hand_suit_card:
                    preferred_card = 'R' + suit
                else:  # count color occurance
                    idx = np.argsort((np.array(all_public_color_nums) + np.array(cur_private_color_nums))[:3])
                    most_occurred_color = COLOR[idx[-1]]
                    if most_occurred_color + suit in hand_suit_card:
                        preferred_card = most_occurred_color + suit
                    else:
                        preferred_card = COLOR[idx[-2]] + suit
                nonpreferred_card = hand_suit_card.copy()
                nonpreferred_card.remove(preferred_card)
                nonpreferred_card = nonpreferred_card[0]
                    
                if all_public_suit_num + cur_private_suit_num + hand_suit_num == 5: # 5 same suit: prefer 2 private, 3 public
                    if 2 <= all_public_suit_num <= 3:
                        if len(cur_private) < 3:  # need to put in private; otherwise, not necessary
                            return preferred_card + '0'
                        else:
                            pass
                    else:
                        if len(cur_public) < 3:  # prefer to put in public
                            return nonpreferred_card + '1'
                        else:
                            return preferred_card + '0'
                    
                if all_public_suit_num + cur_private_suit_num + hand_suit_num == 4: # 4 same suit: prefer 2 private, 2 public
                    if 1 <= all_public_suit_num <= 2:
                        if len(cur_private) < 3:  # need to put in private; otherwise, not necessary
                            return preferred_card + '0'
                        else:
                            pass
                    else:
                        if len(cur_public) < 3:  # prefer to put in public
                            return nonpreferred_card + '1'
                        else:
                            return preferred_card + '0'
        
        # color rules
        # one C/R color in hand, 3-4 in total: immediate score
        for color, hand_color_num, all_public_color_num, cur_private_color_num, hand_color_card in zip(COLOR, hand_color_nums, all_public_color_nums, cur_private_color_nums, hand_color_cards):
            if color in ['G', 'B', 'Y']:
                continue
            if hand_color_num == 1:
                idx = np.argsort((np.array(all_public_suit_nums) + np.array(cur_private_suit_nums)))
                preference = []
                for i in idx[::-1].tolist():
                    if color + SUIT[i] in hand_color_card:
                        preference.append(color + SUIT[i])              
                preferred_card = preference[0]
                
                if all_public_color_num + cur_private_color_num + hand_color_num == 4: # 4 same color: prefer 2 private, 2 public
                    if 2 <= all_public_color_num <= 3:
                        if len(cur_private) < 3:  # need to put in private; otherwise, not necessary
                            return preferred_card + '0'
                        else:
                            pass
                    else:
                        if len(cur_public) < 3:  # prefer to put in public
                            return preferred_card + '1'
                        else:
                            return preferred_card + '0'
                    
                if all_public_color_num + cur_private_color_num + hand_color_num == 3: # 3 same color: prefer 2 private, 1 public
                    if 1 <= all_public_color_num <= 2:
                        if len(cur_private) < 3:  # need to put in private; otherwise, not necessary
                            return preferred_card + '0'
                        else:
                            pass
                    else:
                        if len(cur_public) < 3:  # prefer to put in public
                            return preferred_card + '1'
                        else:
                            return preferred_card + '0' 
                        
        # two C/R colors in hand, 3-4 in total: close to immediate score
        for color, hand_color_num, all_public_color_num, cur_private_color_num, hand_color_card in zip(COLOR, hand_color_nums, all_public_color_nums, cur_private_color_nums, hand_color_cards):
            if color in ['G', 'B', 'Y']:
                continue
            if hand_color_num == 2:
                idx = np.argsort((np.array(all_public_suit_nums) + np.array(cur_private_suit_nums)))
                preference = []
                for i in idx[::-1].tolist():
                    if color + SUIT[i] in hand_color_card:
                        preference.append(color + SUIT[i])              
                preferred_card = preference[0]
                nonpreferred_card = preference[-1]
                
                if all_public_color_num + cur_private_color_num + hand_color_num == 4: # 4 same color: prefer 2 private, 2 public
                    if all_public_color_num == 2:
                        if len(cur_private) < 3:  # need to put in private; otherwise, not necessary
                            return preferred_card + '0'
                        else:
                            pass
                    elif all_public_color_num == 1:
                        if len(cur_private) < 3:  # prefer to put in private
                            return preferred_card + '0'
                        else:
                            return preferred_card + '1'
                    else:
                        if len(cur_public) < 3:  # prefer to put in public
                            return preferred_card + '1'
                        else:
                            return preferred_card + '0'
                    
                if all_public_color_num + cur_private_color_num + hand_color_num == 3: # 3 same color: prefer 2 private, 1 public
                    if len(cur_private) < 3:  # prefer to put in private
                        return preferred_card + '0'
                    else:
                        return preferred_card + '1' 
         
        ## general greedy: far from scoring
        # estimate the value of each card in hand; play the most valuable card in private (if larger than threshold) or public (otherwise)
        
        def estimate_card_value(card, other_cur_hand, all_public, cur_private):
            if card in ['DW', 'XW']:
                return 0 # We do not evaluate joker card as they should have been taken care of in the previous conditions
            
            color, suit = card[0], card[1]
            value = 2 if color in ['C', 'R'] else 1.3
   
            for other_card in all_public:
                if suit in other_card:
                    value += 1
                if color in other_card and color in ['C', 'R']:
                    value += 1
                if color in other_card and color in ['G', 'B', 'Y']:
                    value += 0.3

            for other_card in cur_private:
                if suit in other_card:
                    value += 1.2
                if color in other_card and color in ['C', 'R']:
                    value += 1.2
                if color in other_card and color in ['G', 'B', 'Y']:
                    value += 0.5
                    
            for other_card in other_cur_hand:
                if suit in other_card:
                    value += 0.5
                if color in other_card and color in ['C', 'R']:
                    value += 0.5
                if color in other_card and color in ['G', 'B', 'Y']:
                    value += 0.15
            return value
        
        values = []
        for i, card in enumerate(cur_hand):
            values.append(estimate_card_value(card, cur_hand[:i] + cur_hand[i+1:], all_public, cur_private))
        max_card, max_value = cur_hand[np.argmax(values)], np.max(values)
        
        if cur_turn <= 1:  # first turn always put in private
            return max_card + '0'
        if len(cur_private) == 3: # must put in public
            return max_card + '1'
        if len(cur_public) == 3: # must put in private
            return max_card + '0'
        
        if max_value >= 6:
            return max_card + '0'
        else:
            return max_card + '1'

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
