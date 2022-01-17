from email import policy
import torch
import torch.nn as nn
import numpy as np

import neural_network

def masked_softmax(x, mask, sum_dim):
    exp_x = x.exp()
    zeros_like_x = torch.zeros_like(x)

    masked_exp_x = torch.where(mask, exp_x, zeros_like_x)
    masked_exp_x_sum = masked_exp_x.sum(sum_dim, keepdim=True)

    masked_softmax = torch.where(mask, exp_x / (masked_exp_x_sum + 1e-8), zeros_like_x)
    return masked_softmax

class NeuralNetwork(neural_network.Base):
    def __init__(self, state_shape, action_space, device):
        super().__init__()
        
        self.common = nn.Sequential(
            nn.Conv2d(state_shape[0], 16, (5), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, (3), padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, (3), padding='same'),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten()
        ) 

        self.policy_stream = nn.Sequential(
            nn.Linear(256, 200),
            nn.ReLU(),
            nn.Linear(200, action_space)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(256, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

        self.softmax = nn.Softmax(0)
    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        self.mse = nn.MSELoss()
        
        self.device = device
    
    def forward(self, x, mask):
        x = self.common(x)
        
        value = self.value_stream(x)
        policy = masked_softmax(self.policy_stream(x), mask, 1)
        policy = policy.masked_select(mask)
        
        return value, policy

    def __call__(self, x, mask):
        x = torch.from_numpy(x).to(self.device)
        mask = torch.from_numpy(mask).to(self.device)
        
        value, policy = self.forward(x, mask)
        
        value = value.detach().cpu().numpy()
        policy = policy.detach().cpu().numpy()
        
        return value, policy      
        
        
    def _calc_loss(self, *examples):
        x, mask, value_target, policy_target = [torch.from_numpy(x).to(self.device) for x in examples]
        
        value, policy = self.forward(x, mask)
        
        loss = self.mse(value, value_target) + self.mse(policy, policy_target)
        return loss
        
    def fit(self, examples):
        x, mask, value_target, policy_target = examples
        
        x = np.stack(x)
        mask = np.stack(mask)
        value_target = np.stack(value_target)
        policy_target = np.concatenate(policy_target, 0)
        
        loss = self._calc_loss(x, mask, value_target, policy_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss = float(loss.detach().cpu().numpy())
        return loss