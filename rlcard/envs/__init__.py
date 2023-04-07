''' Register new environments
'''
from rlcard.envs.env import Env
from rlcard.envs.registration import register, make

register(
    env_id='swy-blm',
    entry_point='rlcard.envs.swy_blm:SwyBlmEnv',
)