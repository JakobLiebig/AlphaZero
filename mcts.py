import numpy as np
from functools import partial
from math import sqrt

import game_state
import neural_network

EPSILON = 1e-10

class Node:
    def root(initial_state: game_state.Base):
        return Node(None, None, None, initial_state)
    
    def __init__(self, parent, parent_action, prior_prob, state : game_state.Base):
        self.parent = parent
        self.parent_action = parent_action
        self.prior_prob = prior_prob
        
        self.num_visits = 0
        self.q = 0.
        self.state = state
        self.fully_expanded = False
        
        self.children = []
           
    def _select_child(self, exploration_bias):
        key = partial(Node._calc_ucb, exploration_bias=exploration_bias)
        
        return max(self.children, key=key)
    
    def _calc_ucb(self, exploration_bias):
        n = self.num_visits
        N = self.parent.num_visits
        p = self.prior_prob
        
        u = exploration_bias * p * sqrt(N + EPSILON) / (1 + n)
        
        return self.q + u
    
    def traverse(self, exploration_bias = sqrt(2)):
        cur_node = self
       
        while (cur_node.fully_expanded
               and not cur_node.state.is_terminal()):
            cur_node = cur_node._select_child(exploration_bias)
        
        return cur_node
    
    def expand(self, policy):
        self.fully_expanded = True
        
        for index, action in enumerate(self.state.generate_possible_actions()):
            prior_prob = policy[index]
            
            new_state = self.state.step(action)
            new_child = Node(self, action, prior_prob, new_state)
            
            self.children.append(new_child)
    
    def backprop(self, value):
        cur_node = self
        
        while cur_node != None:
            cur_node.q = (cur_node.num_visits * cur_node.q + value) / (cur_node.num_visits + 1)
            cur_node.num_visits += 1

            cur_node = cur_node.parent
            value = -value
    
    def _get_num_visits(self):
        return self.num_visits 
    
    def policy(self, temperature) -> np.array:
        policy = np.zeros(len(self.children), dtype=np.float32)
        
        for index, child in enumerate(self.children):
            policy[index] = child.num_visits ** temperature
        
        if len(policy) == 0:
            print()
        
        return policy / policy.sum()

class Tree:
    def __init__(self, initial_state: game_state.Base, nn: neural_network.Base):
        self.root = Node.root(initial_state)
        self.nn = nn

    def search(self, iterations: int, temperature: float) -> np.array:
        iterations = iterations - self.root.num_visits
        
        for _ in range(iterations):
            selected_node = self.root.traverse()
            
            if not selected_node.state.is_terminal():
                board_state = selected_node.state.generate_state()[None]
                action_mask = selected_node.state.generate_mask()[None]
                
                value, policy = self.nn(board_state, action_mask)
                
                selected_node.expand(policy)
            else:
                value = selected_node.state.get_reward()
            
            selected_node.backprop(value)
        
        return self.root.policy(temperature)

    def select(self, action):
        for child in self.root.children:
            
            if child.parent_action == action:
                self.root = child
                
                return