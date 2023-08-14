import torch
import torch.nn as nn

class RewardEstimator(nn.Module):
    def __init__(self, state_dim, action_dim, reward_dim, hidden_dim):
        super(RewardEstimator, self).__init__()
        
        self.input_dim = state_dim + action_dim
        self.output_dim = reward_dim
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.output_dim)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        reward = self.fc3(x)
        return reward
