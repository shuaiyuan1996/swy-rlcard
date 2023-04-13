from rlcard.utils.color import bcolor

class HumanAgent(object):
    ''' A human agent for SwyBlm. It can be used to play against trained models
    '''

    def __init__(self):
        ''' Initilize the human agent

        Args:
            num_actions (int): the size of the ouput action space
        '''
        self.use_raw=True
        pass

    @staticmethod
    def step(state):
        ''' Human agent will display the state and make decisions through interfaces

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action decided by human
        '''
        #print(state['raw_obs'])
        _print_state(state['raw_obs'], state['action_record'])
        action = input('>> Enter your action: ')
        while action not in state['raw_legal_actions']:
            action = input('>> Action illegal, please re-enter your action: ')
        _print_action("You", action)
        return action

    def eval_step(self, state):
        ''' Predict the action given the curent state for evaluation. The same to step here.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state), {}
    
def _card_str(card):
    return bcolor.BOLD + bcolor.UNDERLINE + card + bcolor.ENDC
    
def _print_state(state, action_record):
    ''' Print out the state of a given player

    Args:
        player (int): Player id
    '''
    
    if state['cur_turn'] > 0:
        oppo_action = action_record[-1][1]
    else:
        oppo_action = None
    
    your_oppo_string = bcolor.RED + "Your opponent" + bcolor.ENDC
    your_string = bcolor.CYAN + "Your" + bcolor.ENDC
        
    # print last opponent action if applicable
    if oppo_action is not None:
        print('-' * 29 + " " + your_oppo_string + "'s turn " + '-' * 29)
        print("{} drawed a card.".format(your_oppo_string))
        _print_action(your_oppo_string, oppo_action)
     
    # print your turn
    print('-' * 34 + " " + your_string + " turn " + '-' * 35)
    print(bcolor.CYAN + "You" + bcolor.ENDC + " drawed a new card {}.".format(_card_str(state['cur_hand'][-1])))
    print("Current borad:")
    print("  " + your_oppo_string + "'s hand: 3 cards" + ' ' * 15 + your_oppo_string + "'s private: {} cards ".format(state['oppo_private_count']))
    string = "  Board: {}, {}".format(_card_str(state['board_public'][0]), _card_str(state['board_public'][1]))
    string += ' ' * 4 + your_string + " public: {0:10s}".format(', '.join(map(_card_str, state['cur_public'])))
    string += ' ' * 4 + your_oppo_string + "'s public: {0:10s}".format(', '.join(map(_card_str, state['oppo_public'])))
    print(string)
    print("  " + your_string + " hand: {}, {}, {}, {}".format(*map(_card_str, state['cur_hand'])) + ' ' * 19 + your_string + " private: {0:10s}".format(', '.join(map(_card_str, state['cur_private']))))

def _print_action(name, action):
    ''' Print out an action in a nice form

    Args:
        name (str): Player name
        action (str): A string a action
    ''' 
    card, move = action[:2], int(action[2])
    if move == 1:
        print("{} played a public card {}.".format(name, _card_str(card)))
    else:
        if name == "You":
            print("You played a private card {}.".format(_card_str(card)))
        else:
            print("{} played a private card.".format(name))
