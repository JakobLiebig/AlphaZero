import numpy as np
from typing import Tuple
import torch

class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def call(self, x: np.array, mask: np.array) -> Tuple[np.array, np.array]:
        raise NotImplementedError
    
    def forward(self, x, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def fit(self, examples) -> float:
        raise NotImplementedError