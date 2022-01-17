import numpy as np

class Base():
    def step(self, action): # -> Base
        raise NotImplementedError
    
    def game_over(self) -> bool:
        raise NotImplementedError
    
    def get_reward(self) -> float:
        raise NotImplementedError
    
    def generate_state(self) -> np.array:
        raise NotImplementedError
    
    def generate_mask(self) -> np.array:
        raise NotImplementedError
    
    def generate_possible_actions(self) -> np.array:
        raise NotImplementedError