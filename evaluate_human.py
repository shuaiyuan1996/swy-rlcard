''' A toy example of playing against a swy-bm agent
'''
import argparse

import rlcard
from rlcard.agents.human_agent import HumanAgent, _print_action
from rlcard.agents.dmc_agent.model import DMCAgent
from rlcard.games.bailongmen.utils import SUIT, COLOR

from rlcard.utils.color import bcolor

import numpy as np
import torch

parser = argparse.ArgumentParser("Evaluate trained agents by competing against human inputs")
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--ai_agent', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--human_order', type=int, choices=[1, 2], default=None)
args = parser.parse_args()

DEVICE = "cuda:0" if args.cuda else "cpu"

if __name__ == "__main__":

    ## Make environment
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

    # human agent
    human_agent = HumanAgent()

    while (True):
        print('=' * 30 + "     Start Game     " + '=' * 30)
        
        ## Decide who goes first
        if args.human_order is None:  # random toss
            human_order = np.random.randint(1, 3)
        else:
            human_order = args.human_order

        if human_order == 1:
            env.set_agents([human_agent, ai_agents[1]])
            human_player, ai_player = 0, 1
            print("You go first, and AI follows!")
        else:
            env.set_agents([ai_agents[0], human_agent])
            human_player, ai_player = 1, 0
            print("AI goes first, and you follow!")

        trajectories, payoffs = env.run(is_training=False)
        
        def _your_oppo_code(you_oppo):
            return bcolor.RED + you_oppo + bcolor.ENDC
        def _you_code(you):
            return bcolor.CYAN + you + bcolor.ENDC

        
        # If the human does not take the final action, we need to
        # print other players action      
        if human_order == 1:
            oppo_action = trajectories[0][-1]['action_record'][-1][1]
            print('-' * 29 + " " + _your_oppo_code("Your opponent") + "'s turn " + '-' * 29)
            _print_action(_your_oppo_code("Your opponent"), oppo_action)

        # print final result
        print('=' * 30 + "    Final Result    " + '=' * 30)
        print("Final table: (o - public; " + _you_code("x - your private") + "; " + _your_oppo_code("* - your opponent's private") + ")")
        print("      " + "  ".join(COLOR))
        
        for s in range(4):
            string = "  {}: ".format(SUIT[s])
            for c in range(5):
                cur_mark = env.game.final_tables[human_player][0][c, s]
                oppo_mark = env.game.final_tables[ai_player][0][c, s]
            
                if cur_mark == 0 and oppo_mark == 0:
                    string += "   "
                elif cur_mark == 0 and oppo_mark == 1:
                    string += _your_oppo_code(" * ")
                elif cur_mark == 1 and oppo_mark == 0:
                    string += _you_code(" x ")
                elif cur_mark == 1 and oppo_mark == 1:
                    string += " o "
            print(string)
        print("\n      XW  DW")
        string = "      "
        for i in range(2):
            cur_mark = env.game.final_tables[human_player][1][i]
            oppo_mark = env.game.final_tables[ai_player][1][i]          
            if cur_mark == 0 and oppo_mark == 0:
                string += "   "
            elif cur_mark == 0 and oppo_mark == 1:
                string += _your_oppo_code(" * ")
            elif cur_mark == 1 and oppo_mark == 0:
                string += _you_code(" x ")
            elif cur_mark == 1 and oppo_mark == 1:
                string += " o "
        print(string)
        
        print(_you_code("Your") + " score = {}. ".format(env.game.final_score[human_player]) + _your_oppo_code("Your opponent") + "'s score = {}.".format(env.game.final_score[ai_player]))
        
        if env.game.final_score[human_player] > env.game.final_score[ai_player]:
            print(_you_code("You") + ' win!')
        else:
            print(_you_code("You") + ' lose...')

        input("Press <Enter> to start another game...")
