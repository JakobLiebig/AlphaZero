import numpy as np
import random
from collections import deque

import logger
import game_state
import neural_network

import mcts

class ReplayBuffer:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)
    
    def extend(self, states, masks, values, policys):
        self.memory.extend(zip(states, masks, values, policys))
        
    def sample(self, size):
        sample = random.sample(self.memory, size)
        states, masks, values, policys = [np.stack(x) for x in zip(*sample)]
        
        return states, masks, values, policys

class Coach():
    def __init__(self, initial_state: game_state.Base, buffer_size, logger: logger.Base = None):
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.inital_state = initial_state
        self.logger = logger
    
    def _try_log(self, *args):
        
        if not self.logger is None:
            self.logger.log(args)
    
    def _self_play(self, nn: neural_network.Base, temperature, search_iterations):
        current_state = self.inital_state
        tree = mcts.Tree(current_state, nn)
        
        states = []
        action_masks = []
        policys = []
        
        while not current_state.is_terminal():
            policy = tree.search(search_iterations, temperature)
            action = np.random.choice(current_state.generate_possible_actions(), p=policy)
            
            states.append(current_state.generate_state())
            action_masks.append(current_state.generate_action_mask())
            policys.append(policy)
            
            current_state = current_state.step(action)
            tree.select(action)
        
        values = []
        
        outcome = current_state.get_reward()
        for _ in states:
            values.append(outcome)
            
            outcome = -outcome
        
        return states, action_masks, policys, values[::-1]
    
    def train(self, nn: neural_network.Base, iterations, batch_size, temperature, search_iterations):
        self._try_log(type='event start training', message='')
        
        for i in range(iterations):
            transitions = self._self_play(nn, temperature, search_iterations)
            self.replay_buffer.append(transitions)

            sample = self.replay_buffer.sample(batch_size)
            loss = nn.fit(sample)
            
            self._try_log(type='loss', message=loss)
            self._try_log(type='progress training', message=i / iterations)
        
        self._try_log(type='event stop training', message='')
        
        
    def _play(self, contestants, scores, temperature):
        current_state = self.inital_state
        active_player = random.choice([0, 1])
        
        while not current_state.is_terminal:
            policy = contestants[active_player].policy(temperature)
            action = np.random.choice(current_state.generate_possible_actions(), p=policy)
            
            active_player = 0 if active_player == 1 else 1
            
            current_state = current_state.step(action)
            [contestant.select(action) for contestant in contestants]
            
        outcome = current_state.get_reward()
        scores[active_player] = -outcome
        
    def pit(self, nns, iterations, temperature):
        scores = np.zeros(2)
        contestants = [mcts.Tree(self.inital_state, nn) for nn in nns]
        
        self._try_log(type='event start evaluation', message='')
             
        for i in range(iterations):
            self._play(contestants, scores, temperature)
            
            self._try_log(type='progress evaluation', message=i / iterations)
        
        winner = nns[scores.argmax()]
        
        self._try_log(type='event stop evaluation', message=f'winner= {str(winner)}')

        return winner, scores