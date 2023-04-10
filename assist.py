''' Usinig a trained model to assist decision-making (outer environment)
'''
import os
import argparse

import numpy as np
import torch

import rlcard
from rlcard.agents.dmc_agent.model import DMCAgent
from rlcard.games.bailongmen.utils import DECK, ID_2_ACTION
from rlcard.games.bailongmen.utils import check_cards_validity

from rlcard.utils.color import bold, color_you, color_your_oppo

display_order = ['GL', 'BL', 'YL', 'CL', 'RL',
                 'GZ', 'BZ', 'YZ', 'CZ', 'RZ',
                 'GJ', 'BJ', 'YJ', 'CJ', 'RJ',
                 'GM', 'BM', 'YM', 'CM', 'RM',
                 'XW', 'DW']
card_to_order = {'GL': 0, 'BL': 1, 'YL': 2, 'CL': 3, 'RL': 4,
                 'GZ': 5, 'BZ': 6, 'YZ': 7, 'CZ': 8, 'RZ': 9,
                 'GJ': 10, 'BJ': 11, 'YJ': 12, 'CJ': 13, 'RJ': 14,
                 'GM': 15, 'BM': 16, 'YM': 17, 'CM': 18, 'RM': 19,
                 'XW': 20, 'DW': 21}
    
def sort_cards(cards):
    order = list(map(lambda x: card_to_order[x], cards))
    return np.array(cards)[np.argsort(order)].tolist()

        
@torch.no_grad()
def assist_game(env, agents):
    
    try:
        raw_obs = {}

        # enter initial info
        board_public = input(">> Enter the 2 initial public cards on board: ").split()
        valid, unrecognized_card = check_cards_validity(board_public)
        while not valid or len(board_public) != 2:
            if not valid:
                board_public = input(">> Card '{}' not recognized. Please re-enter all cards again: ".format(bold(unrecognized_card))).split()
            elif len(board_public) != 2:
                board_public = input(">> Please enter exactly 2 cards. Please re-enter all cards again: ").split()
            valid, unrecognized_card = check_cards_validity(board_public)

        raw_obs['board_public'] = board_public
        print("Starting with board public cards {} and {}.".format(*bold(raw_obs['board_public'])))

        # enter who to go first
        print('=' * 30 + "     Start Game     " + '=' * 30)
        order = input(">> Are {} playing first-hand (F) or second-hand (S)? ".format(color_you("you")))
        while order not in ['F', 'S']:
            order = input(">> Invalid input. Please enter 'F' or 'S': ")

        if order == 'F':
            agent = agents[0]
            raw_obs['player_id'] = 0
            print("OK! First-hand agent enabled.")
        elif order == 'S':
            agent = agents[1]
            raw_obs['player_id'] = 1
            print("OK! Second-hand agent enabled.")

        # TODO: decide how to switch initial hand cards: maybe?
        pass

        # enter starting hand
        cur_hand = input(">> Enter {} 3 initial cards in hand: ".format(color_you("your"))).split()
        valid, unrecognized_card = check_cards_validity(cur_hand)
        while not valid or len(cur_hand) != 3:
            if not valid:
                cur_hand = input(">> Card '{}' not recognized. Please re-enter all cards again: ".format(bold(unrecognized_card))).split()
            elif len(cur_hand) != 3:
                cur_hand = input(">> Please enter exactly 3 cards. Please re-enter all cards again: ").split()
            valid, unrecognized_card = check_cards_validity(cur_hand)

        cur_hand = sort_cards(cur_hand)
        raw_obs['cur_hand'] = cur_hand
        print("Starting with hand cards {}, {}, {}.".format(*bold(raw_obs['cur_hand'])))

        # start playing!
        turn = 0
        raw_obs['cur_public'] = []
        raw_obs['cur_private'] = []
        raw_obs['oppo_public'] = []
        raw_obs['oppo_private_count'] = 0

        while turn < 12:
            # opponent turn
            if not (turn == 0 and raw_obs['player_id'] == 0):
                print('-' * 29 + " {}'s turn ".format(color_your_oppo("Your opponent")) + '-' * 29)
                oppo_action = input(">> Enter {}'s action: card name if they played it public, '?' if they played it private: ".format(color_your_oppo("your opponent"))).strip()
                while oppo_action not in DECK and oppo_action != '?':
                    oppo_action = input(">> Unrecognized {} action {}. Please re-enter: ".format(color_your_oppo("opponent"), oppo_action)).strip()

                if oppo_action == '?':
                    print("{} played a private card.".format(color_your_oppo("Your opponent")))
                    raw_obs['oppo_private_count'] += 1
                else:
                    print("{} played a public card {}.".format(color_your_oppo("Your opponent"), bold(oppo_action)))
                    raw_obs['oppo_public'].append(oppo_action)

                turn += 1

            # your turn
            print('-' * 34 + " {} turn ".format(color_you("Your")) + '-' * 35)
            raw_obs['cur_turn'] = turn

            drawed_card = input(">> Enter the new card that {} draw (other than {}, {}, {}): ".format(color_you("you"), *bold(raw_obs['cur_hand']))).strip()
            while drawed_card not in DECK or drawed_card in raw_obs['cur_hand']:
                drawed_card = input(">> Invalid input {}. Please re-enter: ".format(drawed_card)).strip()
            raw_obs['cur_hand'].append(drawed_card)
            raw_obs['cur_hand'] = sort_cards(raw_obs['cur_hand'])

            # make decision
            state = env._extract_state(raw_obs)
            action_id = agent.step(state)
            action = ID_2_ACTION[action_id]

            card, sec = action[:2], "public" if action[2] == '1' else "private"
            input("{} action: please put {} in the {} area and press <ENTER> to proceed...".format(color_you("Your"), bold(card), bold(sec)))
            raw_obs['cur_' + sec].append(card)
            raw_obs['cur_hand'].remove(card)

            turn += 1

            if turn == 11:
                break  # no need to enter the opponent's last action

        print('=' * 29 + "     End of Game     " + '=' * 30)
        return
    
    except Exception as e:
        # keyboard interrupt to jump in (e.g., for debugging)
        import IPython; IPython.embed(); exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate agent against baselines")
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--ai_agent', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    DEVICE = "cuda:0" if args.cuda else "cpu"
    
    ## Make environment (for info only; not used for gaming)
    env = rlcard.make('swy-blm')
    
    ## Prepare agents
    if args.ai_agent == 'dmc':
        ai_agents = []
        for p in range(env.num_players):
            ai_agents.append(DMCAgent(env.state_shape[p], env.action_shape[p], mlp_layers=[256, 256, 256, 128], device=DEVICE))

    # load trained ai agents
    ckpt_states = torch.load(args.model_path, map_location=DEVICE)
    for p in range(env.num_players):
        ai_agents[p].load_state_dict(ckpt_states['model_state_dict'][p])

    print("AI model version: {} at {}".format(ai_agents[0].__class__, args.model_path))
    
    while True:
        assist_game(env, ai_agents)
        input(">> Press <Enter> to start another game...")