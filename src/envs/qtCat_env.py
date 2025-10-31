from dataclasses import dataclass, field
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from src.envs.utils import area, bounce
from collections import deque

@dataclass
class qtCatConfig:
    n : any = 13 # semi-length
    
    @property
    def intitial_state(self): 
        output = deque([2] * (2 * self.n), maxlen=2 * self.n)  # padding token is 2
        return output

    @property
    def horizon_length(self):
        return self.n * 2
    
class qtCatEnv(Env):
    def __init__(self, input, config: qtCatConfig = qtCatConfig()):
        self.config = config
        self.action_space = Discrete(2)  # two actions: 0 or 1
        self.observation_space = Box(low=0, high=2, shape=(2 * self.config.n,), dtype=np.int8)
        self.current_step = 0
        
        self.input = input

        self.state = self.config.intitial_state
        self.input_area, self.input_bounce = area(self.state[0]), bounce(self.state[0])
        self.target_area, self.target_bounce = self.input_bounce, self.input_area

    def step(self, action):
        self.current_step += 1

        self.state.appendleft(action)

        done = self.current_step >= self.config.horizon_length

        self.cur_area, self.cur_bounce = area(self.state), bounce(self.state)
        fail = self.cur_area > self.target_area or self.cur_bounce > self.target_bounce
        if not fail:
            area_err_reward = 1 - abs(self.cur_area - self.target_area) / (self.config.n * 2)
            bounce_err_reward = 1 - abs(self.cur_bounce - self.target_bounce) / (self.config.n * 2)
        reward = fail * -2 + area_err_reward + bounce_err_reward

        return (
            self.state,
            reward,
            done,
        )
        
    def reset(self, input):
        self.state = self.state = self.config.intitial_state
        self.current_step = 0
        self.input_area, self.input_bounce = area(self.state[0]), bounce(self.state[0])
        self.target_area, self.target_bounce = self.input_bounce, self.input_area

        return self.state
    
    def render(self):
        pass