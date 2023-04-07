''' A toy example of playing against a swy-bm agent
'''
import sys
sys.path.append('/home/users/shuai/code/swy-rlcard')

import rlcard
from rlcard import models
from rlcard.agents.human_agents.swy_blm_human_agent import HumanAgent, _print_action
from rlcard.agents.dmc_agents.model import DMCAgent

# Make environment
env = rlcard.make('swy-blm')
human_agent = HumanAgent(env.num_actions)
dmc_agent = DMCAgent(                    self.env.state_shape,   ///////
                    self.action_shape,
                    mlp_layers=mlp_layers,
                    exp_epsilon=self.exp_epsilon,
                    device=str(device))

checkpoint_states = torch.load(
            self.checkpointpath,
            map_location="cuda:"+str(self.training_device) if self.training_device != "cpu" else "cpu"
    )


env.set_agents([
    human_agent,
    cfr_agent,
])

print(">> UNO rule model V1")

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state['current_player']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses ', end='')
        _print_action(pair[1])
        print('')

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win!')
    else:
        print('You lose!')
    print('')
    input("Press any key to continue...")
