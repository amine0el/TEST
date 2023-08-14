import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mo_gymnasium as mo_gym


class DQN(nn.Module):
    def __init__(self, obs_space, action_space):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, env):
        self.env = env
        self.model = DQN(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.eps = 0.5
        self.eps_decay = 0.995

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        if np.random.rand() < self.eps:
            action = np.random.randint(self.env.action_space.n)
        else:
            q_values = self.model(state)
            action = torch.argmax(q_values).item()
        return action

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        action = torch.LongTensor([action])

        current_q_value = self.model(state)[0][action]
        next_q_value = self.model(next_state).max(1)[0].detach()
        target_q_value = reward + 0.99 * next_q_value * (1 - done)

        loss = self.criterion(current_q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if done:
            self.eps *= self.eps_decay

def train(agent, episodes):
    episode_rewards = []
    for episode in range(episodes):
        state = agent.env.reset()[0]
        done = False
        episode_reward = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, absorbing, _ = agent.env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
        episode_rewards.append(episode_reward)
        if episode % 10 == 0:
            print(f'Episode: {episode}, Reward: {episode_reward}, Epsilon: {agent.eps}')
    return episode_rewards

env = mo_gym.make("deep-sea-treasure-v0")
agent = Agent(env)
rewards = train(agent, 500)
