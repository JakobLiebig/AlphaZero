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
    
    def __len__(self):
        return len(self.memory)
    
    def extend(self, states, masks, values, policys):
        self.memory.extend(zip(states, masks, values, policys))
        
    def sample(self, size):
        sample = random.sample(self.memory, size)
        states, masks, values, policys = zip(*sample)
        
        return states, masks, values, policys

class Coach():
    def __init__(self, initial_state: game_state.Base, buffer_size, logger: logger.Base = None):
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.inital_state = initial_state
        self.logger = logger
    
    def _try_log(self, *args):
        
        if not self.logger is None:
            self.logger.log(*args)
    
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
            action_masks.append(current_state.generate_mask())
            policys.append(policy)
            
            current_state = current_state.step(action)
            tree.select(action)
        
        values = []
        
        outcome = current_state.get_reward()
        for _ in states:
            values.append(outcome)
            
            outcome = -outcome
        
        return states, action_masks, values[::-1], policys
    
    def train(self, nn: neural_network.Base, iterations, batch_size, temperature, search_iterations):
        self._try_log('event start training', '')
        
        for i in range(iterations):
            transitions = self._self_play(nn, temperature, search_iterations)
            self.replay_buffer.extend(*transitions)

            if len(self.replay_buffer) >= batch_size:
                sample = self.replay_buffer.sample(batch_size)
                loss = nn.fit(sample)
            
                self._try_log('loss', loss)
            self._try_log('progress training', i / iterations)
        
        self._try_log('event stop training', '')
        
        
    def _play(self, nns, scores, search_iterations, temperature):
        current_state = self.inital_state
        active_player = 0
        trees = []
        
        while not current_state.is_terminal():
            if len(trees) != 2:
                trees.append(mcts.Tree(current_state, nns[len(trees)]))
            
            policy = trees[active_player].search(search_iterations, temperature)
            action = np.random.choice(current_state.generate_possible_actions(), p=policy)
            
            active_player = 0 if active_player == 1 else 1
            
            current_state = current_state.step(action)
            [tree.select(action) for tree in trees]
            
        outcome = current_state.get_reward()
        scores[active_player] = -outcome
        
    def pit(self, nns, iterations, temperature, search_iterations):
        scores = np.zeros(2)
        
        self._try_log('event start evaluation', '')
             
        for i in range(iterations):
            self._play(nns, scores, search_iterations, temperature)
            
            self._try_log('progress evaluation', i / iterations)
        
        winner = nns[scores.argmax()]
        
        self._try_log('event stop evaluation', f'winner= {str(winner)}')

        return winner, scores