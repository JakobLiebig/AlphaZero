from copy import deepcopy
import torch
import os, sys

import_path = os.path.abspath('.')
sys.path.insert(1, import_path)

from vg_neural_network import NeuralNetwork
from vg_game_state import GameState
from vg_logger import Logger

import logger
import coach

initial_state = GameState.initial(8, 8)

logger_ = Logger()
trainer = coach.Coach(initial_state, 1000, logger_)

batch_size = 1

mcts_iterations = 1
train_temperature = 2.
eval_temperature = 4.

iterations = 1
train_iterations = 1
eval_iterations = 1


current_best = NeuralNetwork([2, 8, 8], 8, torch.device('cpu'))

for _ in range(iterations):
    new_contestant = deepcopy(current_best)
    trainer.train(new_contestant, train_iterations, batch_size, train_temperature, mcts_iterations)
    
    current_best = trainer.pit([current_best, new_contestant], eval_iterations)