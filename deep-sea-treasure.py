import mo_gymnasium as mo_gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple

# Define the Policy network structure
class PolicyNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, action_size=4):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

# Initialize the environment and the policy
env = mo_gym.make('deep-sea-treasure-v0')
policy_net = PolicyNetwork(env.observation_space.shape[0], 128, env.action_space.n)

optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

# Placeholder for a single transition in our environment
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Placeholder for pareto front
pareto_front = []

def get_action(state):
    # Extract the grid position and convert it to a tensor
    grid_position = np.array(state)  # assuming the first two elements are the coordinates
    grid_position = torch.from_numpy(grid_position).float()
    
    # TODO: Process the remaining state information if needed
    
    probs = policy_net(grid_position)
    action = np.random.choice(np.array([i for i in range(env.action_space.n)]), p=probs.detach().numpy())
    return action


def update_policy(batch):
    states = torch.stack([torch.from_numpy(np.array(t.state)).float() for t in batch])
    actions = torch.stack([torch.tensor(t.action) for t in batch])
    rewards = torch.stack([torch.tensor(sum(t.reward)) for t in batch])

    Gt = 0
    returns = []
    for r in rewards.flip(dims=(0,)):
        Gt = r + 0.99 * Gt
        returns.insert(0, Gt)
    returns = torch.tensor(returns)

    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps)

    log_probs = torch.log(policy_net(states)).gather(1, actions.view(-1, 1)).view(-1)
    loss = - (log_probs * returns).sum()

    optimizer.zero_grad()
    loss.backward()

    # clip the gradients
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

    optimizer.step()


def is_pareto_efficient(costs):
    # This part is taken from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1) 
    return is_efficient

# The main training loop
for i_episode in range(1000):  # Assuming we're going for 1000 episodes
    state = env.reset()
    state = state[0]
    transitions = []
    for t in range(100):  # Assuming a maximum of 100 timesteps per episode
        action = get_action(state)

        next_state, reward, done, absorbing, _ = env.step(action)

        # Store the transition in our batch
        transitions.append(Transition(state, action, next_state, reward))

        if done:
            break

        state = next_state

    # TO-DO: replace reward with your multi-objective reward here.
    rewards = np.array([t.reward for t in transitions])
    #print(rewards)
    if len(pareto_front) == 0:
        pareto_front = rewards
    else:
        pareto_front = np.concatenate((pareto_front, rewards), axis=0)

    pareto_efficient = is_pareto_efficient(pareto_front)
    pareto_front = pareto_front[pareto_efficient]
    #print(pareto_front)
    update_policy(transitions)
