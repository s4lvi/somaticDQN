import random
from stable_baselines3.common.env_checker import check_env
import numpy as np

from gym import Env, spaces

"""
Implementation of the Iowa Gambling Task for use in reinforcement learning agents.
by JM Salvi
"""

class IGT(Env):

    def __init__(self):
        super(IGT, self).__init__()
        self.agent_bank = 2000
        self.cards = 100
        self.action_space = spaces.Discrete(4)
        high = np.array(
            [
                5000,
                100
            ],
            dtype=int,
        )
        low = np.array(
            [
                -5000,
                0
            ],
            dtype=int,
        )
        self.observation_space = spaces.Box(high=high, low=low, dtype=int)
        return

    def _get_obs(self):
        return np.array([self.agent_bank, self.cards],dtype=int)

    def reset(self):
        self.agent_bank = 2000
        self.cards = 100
        return self._get_obs()

    def step(self, action):
        reward = 0
        self.cards -= 1
        if (action == 0 or action == 1):
            self.agent_bank += 100
            reward = 1
            if (random.randint(0,1)):
                self.agent_bank -= 150
                reward = -1
        elif (action == 2 or action == 3):
            self.agent_bank += 50
            reward = 1
            if (random.randint(0,1)):
                reward = 0
        else: 
            raise ValueError("Received invalid action={} which is not part of the action space".format(action)) 

        done = bool(self.agent_bank <= 0 or self.cards == 0)
        info = {}
        return self._get_obs(), reward, done, info
    
    def render(self):
        print(self.agent_bank)
