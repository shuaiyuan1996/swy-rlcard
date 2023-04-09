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

COLOR = ["G", "B", "Y", "C", "R"]

SUIT = ["M", "L", "Z", "J"]

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

def compute_table_score(table, joker):
    score = 0

    for suit_id in range(4):  # 4, 8 points for Mei, Lan, Zhu, and Ju
        suit_count = table[:, suit_id].sum()
        score += (suit_count == 5) * 8 + (suit_count == 4) * 4

    for color_id in range(3):  # 1, 3 points for Grey, Blue, and Yellow
        color_count = table[color_id, :].sum()
        score += (color_count == 4) * 3 + (color_count == 3) * 1

    for color_id in range(3, 5):  # 3, 6 points for Colorful, and Red
        color_count = table[color_id, :].sum()
        score += (color_count == 4) * 6 + (color_count == 3) * 3

    score += (joker.sum() == 2) * 5  # 5 points for two jokers
    
    return score

def count_suit(cards, suit):
    return np.sum([suit in card for card in cards])

def count_color(cards, color):
    return np.sum([color in card for card in cards])
    
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
        