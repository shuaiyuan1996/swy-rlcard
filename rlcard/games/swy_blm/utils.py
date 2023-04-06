''' Swy-Blm utils
'''
import os
import numpy as np
import json
from collections import OrderedDict
import threading
import collections

import rlcard

# Read required docs
ROOT_PATH = rlcard.__path__[0]

# Deck
DECK = ['GM', 'GL', 'GZ', 'GJ',
        'BM', 'BL', 'BZ', 'BJ',
        'YM', 'YL', 'YZ', 'YJ',
        'CM', 'CL', 'CZ', 'CJ',
        'RM', 'RL', 'RZ', 'RJ',
        'XW', 'DW']

# Action space
ID_2_ACTION = [card + '0' for card in DECK] + [card + '1' for card in DECK]
ACTION_2_ID = {}
for i, action in enumerate(ID_2_ACTION):
    ACTION_2_ID[action] = i

def _cards2array(cards):
    array = np.zeros(len(DECK), dtype=np.float32)
    for i, deck_card in enumerate(DECK):
        if deck_card in cards:
            array[i] = 1.
    return array

def _cards2table(cards):
    ''' Compute the table for scores
    '''
    array = _cards2array(cards)
    regular = array[:20].reshape((5, 4))
    joker = array[20:]
    return regular, joker

    
# Dealer
class SwyBlmDealer:
    def __init__(self, np_random):
        self.deck = DECK.copy()
        np_random.shuffle(self.deck)
        
        self.top = 0  # pointing to the top card in the deck
        
    def draw_cards(self, n=1):        
        cards = self.deck[self.top:(self.top + n)]
        self.top += n 
        return cards
        