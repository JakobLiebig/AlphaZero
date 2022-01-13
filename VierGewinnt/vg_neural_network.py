from turtle import forward
import torch
import torch.nn as nn

import neural_network

class NeuralNetwork(neural_network.Base):
    def __init__(self, state_shape, action_space, device):
        super().__init__()
        
        self.common = nn.Sequential(
            nn.Conv2d(state_shape[0], 16, (5), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, (4), padding='same'),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(8, 4, (3), padding='same'),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten()
        ) 

        self.policy_stream = nn.Sequential(
            nn.Linear(..., 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(..., 200),
            nn.ReLU(),
            nn.Linear(200, action_space)
        )

        self.softmax = nn.Softmax(1)
    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        self.mse = nn.MSELoss()
        
        self.device = device
    
    def forward(self, x, mask):
        x = self.common(x)
        
        value = self.value_stream(x)
        policy = self.policy_stream(x).masked_select(mask)
        policy = self.softmax(policy)
        
        return value, policy

    def __call__(self, x, mask):
        x = torch.from_numpy(x, device=self.device)
        mask = torch.from_numpy(mask, device=self.device)
        
        value, policy = self.forward(x, mask)
        
        value = value.cpu().numpy()
        policy = policy.cpu().numpy()
        
        return value, policy      
        
        
    def _calc_loss(self, examples):
        x, mask, value_target, policy_target = [torch.from_numpy(x, device=self.device) for x in examples]
        
        value, policy = self.forward(x, mask)
        
        loss = self.mse(value, value_target) + self.mse(policy, policy_target)
        return loss
        
    def fit(self, examples):
        loss = self._calc_loss(examples)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss = float(loss.cpu().numpy())
        return loss