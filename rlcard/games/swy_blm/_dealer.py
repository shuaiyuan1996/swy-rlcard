# -*- coding: utf-8 -*-
''' Implement SwyBlm Dealer class
'''
import functools

from rlcard.games.doudizhu.utils import cards2str, doudizhu_sort_card

class SwyBlmDealer:
    ''' Dealer will shuffle, deal cards, and determine players' roles
    '''
    def __init__(self, np_random):
        '''Give dealer the deck
        '''
        
        self.deck = []
        for color in ['G', 'B', 'Y', 'C', 'R']:
            for suit in ['M', 'L', 'Z', 'J']:
                self.deck.append(color + suit)
        self.deck += ['XW', 'DW']
        np_random.shuffle(self.deck)
        
        self.top_ptr = 0  # pointing to the top card in the deck
        

    def draw_card(self, n=1):
        ''' Draw cards
        '''
        
        if self.top_ptr + n > len(self.deck):
            return None
        
        cards =  self.deck[self.top_ptr:(self.top_ptr + n)]
        self.top_ptr += n 
            
        return cards
        
