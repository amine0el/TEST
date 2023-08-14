import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mo_gymnasium as mo_gym
from pymoo.indicators.hv import HV
import numpy.typing as npt
from typing import Callable, List

def hypervolume(ref_point: np.ndarray, points: List[npt.ArrayLike]) -> float:
    """Computes the hypervolume metric for a set of points (value vectors) and a reference point (from Pymoo).

    Args:
        ref_point (np.ndarray): Reference point
        points (List[np.ndarray]): List of value vectors

    Returns:
        float: Hypervolume metric
    """
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)

def get_non_dominated(candidates: set) -> set:
    """This function returns the non-dominated subset of elements.

    Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    The code provided in all the stackoverflow answers is wrong. Important changes have been made in this function.

    Args:
        candidates: The input set of candidate vectors.

    Returns:
        The non-dominated subset of this input set.
    """
    candidates = np.array(list(candidates))  # Turn the input set into a numpy array.
    candidates = candidates[candidates.sum(1).argsort()[::-1]]  # Sort candidates by decreasing sum of coordinates.
    for i in range(candidates.shape[0]):  # Process each point in turn.
        n = candidates.shape[0]  # Check current size of the candidates.
        if i >= n:  # If we've eliminated everything up until this size we stop.
            break
        non_dominated = np.ones(candidates.shape[0], dtype=bool)  # Initialize a boolean mask for undominated points.
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        non_dominated[i + 1 :] = np.any(candidates[i + 1 :] > candidates[i], axis=1)
        candidates = candidates[non_dominated]  # Grab only the non-dominated vectors using the generated bitmask.

    non_dominated = set()
    for candidate in candidates:
        non_dominated.add(tuple(candidate))  # Add the non dominated vectors to a set again.

    return non_dominated



# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, state, action, reward, next_state, done):
#         state = np.expand_dims(state, 0)
#         next_state = np.expand_dims(next_state, 0)

#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
#         return np.concatenate(state), action, reward, np.concatenate(next_state), done

#     def __len__(self):
#         return len(self.buffer)

# class DQN(nn.Module):
#     def __init__(self, obs_space, action_space, num_objectives):
#         super(DQN, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_space, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, action_space * num_objectives)
#         )
#         self.weights = torch.ones(num_objectives) / num_objectives  # equally weighted for simplicity
#         self.action_space = action_space
#         self.num_objectives = num_objectives

#     def forward(self, x):
#         q_values = self.net(x)
#         q_values = q_values.view(-1, self.action_space, self.num_objectives)  # shape: (batch_size, action_space, num_objectives)
#         weighted_q_values = (q_values * self.weights).sum(dim=-1)  # weighted sum across objectives
#         return weighted_q_values  # shape: (batch_size, action_space)



# Updated Agent class with ReplayBuffer
class Agent:
    def __init__(self, env, buffer_size=1000, batch_size=32):
        self.env = env
        # self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        # self.model = DQN(env.observation_space.shape[0], env.action_space.n, env.reward_space.shape[0])
        low_bound = self.env.observation_space.low
        high_bound = self.env.observation_space.high
        self.env_shape = (high_bound[0] - low_bound[0] + 1, high_bound[1] - low_bound[1] + 1)
        # self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.eps = 1.0
        self.eps_decay = .999
        self.global_step = 0
        self.num_objectives = env.reward_space.shape[0]
        self.num_states = np.prod(self.env_shape)
        self.num_actions = env.action_space.n
        self.gamma = 0.95
        self.epsilon = 1.0
        self.seed = 1
        self.np_random = np.random.default_rng(self.seed)
        self.counts = np.zeros((self.num_states, self.num_actions))
        self.non_dominated = [
            [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)] for _ in range(self.num_states)
        ]
        self.avg_reward = np.zeros((self.num_states, self.num_actions, self.num_objectives))

    def score_hypervolume(self, state: int):
        """Compute the action scores based upon the hypervolume metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        action_scores = [hypervolume(self.ref_point, list(q_set)) for q_set in q_sets]
        return action_scores
    
    def get_action(self, state):
        if self.np_random.uniform(0, 1) < self.epsilon:
            return self.np_random.integers(self.num_actions)
        else:
            action_scores = self.score_hypervolume(state)
            return self.np_random.choice(np.argwhere(action_scores == np.max(action_scores)).flatten())

    # def update(self):
    #     if len(self.buffer) < self.batch_size:
    #         return
    #     state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

    #     state = torch.FloatTensor(state)
    #     next_state = torch.FloatTensor(next_state)
    #     reward = torch.FloatTensor(reward)  # assume reward is now a matrix of size batch_size x num_objectives
    #     action = torch.LongTensor(action).unsqueeze(1)  # Shape: (batch_size, 1)
    #     action = action.expand(-1, self.num_objectives)
    #     done = torch.LongTensor(done).unsqueeze(-1)
    #     done = done.expand(-1, self.num_objectives)

    #     current_q_values = self.model(state).gather(1, action)
    #     next_q_values = self.model(next_state).max(1)[0]
    #     target_q_values = reward + self.gamma * next_q_values.unsqueeze(1) * (1 - done)

    #     loss = self.criterion(current_q_values, target_q_values)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.eps *= self.eps_decay


    
    def calc_non_dominated(self, state: int):
        """Get the non-dominated vectors in a given state.

        Args:
            state (int): The current state.

        Returns:
            Set: A set of Pareto non-dominated vectors.
        """
        candidates = set().union(*[self.get_q_set(state, action) for action in range(self.num_actions)])
        non_dominated = get_non_dominated(candidates)
        return non_dominated
    
    def get_q_set(self, state: int, action: int):
        """Compute the Q-set for a given state-action pair.

        Args:
            state (int): The current state.
            action (int): The action.

        Returns:
            A set of Q vectors.
        """
        nd_array = np.array(list(self.non_dominated[state][action]))
        q_array = self.avg_reward[state, action] + self.gamma * nd_array
        return {tuple(vec) for vec in q_array}
    
    def play(self, state):
        action = self.get_action(state)
        next_state, reward, done, absorbing, _ = self.env.step(action)
        self.global_step += 1
        next_state = int(np.ravel_multi_index(next_state, self.env_shape))
        self.counts[state, action] += 1
        self.non_dominated[state][action] = self.calc_non_dominated(next_state)
        self.avg_reward[state, action] += (reward - self.avg_reward[state, action]) / self.counts[state, action]
        state = next_state
        if self.global_step % 100 == 0:
            pf = self._eval_all_policies(self.env)
        return next_state, reward, done, absorbing
    
    def _eval_all_policies(self, env) -> List[np.ndarray]:
        """Evaluate all learned policies by tracking them."""
        pf = []
        for vec in self.get_local_pcs(state=0):
            pf.append(self.track_policy(vec, env))

        return pf

    def track_policy(self, vec, env, tol=1e-3):
        """Track a policy from its return vector.

        Args:
            vec (array_like): The return vector to track.
            env (gym.Env): The environment to track the policy in.
            tol (float, optional): The tolerance for the return vector. (Default value = 1e-3)
        """
        target = np.array(vec)
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_rew = np.zeros(self.num_objectives)
        current_gamma = 1.0

        while not (terminated or truncated):
            state = np.ravel_multi_index(state, self.env_shape)
            closest_dist = np.inf
            closest_action = 0
            found_action = False
            new_target = target

            for action in range(self.num_actions):
                im_rew = self.avg_reward[state, action]
                non_dominated_set = self.non_dominated[state][action]

                for q in non_dominated_set:
                    q = np.array(q)
                    dist = np.sum(np.abs(self.gamma * q + im_rew - target))
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_action = action
                        new_target = q

                        if dist < tol:
                            found_action = True
                            break

                if found_action:
                    break

            state, reward, terminated, truncated, _ = env.step(closest_action)
            total_rew += current_gamma * reward
            current_gamma *= self.gamma
            target = new_target

        return total_rew

    def get_local_pcs(self, state: int = 0):
        """Collect the local PCS in a given state.

        Args:
            state (int): The state to get a local PCS for. (Default value = 0)

        Returns:
            Set: A set of Pareto optimal vectors.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        return get_non_dominated(candidates)


def train(agent, episodes):
    low_bound = env.observation_space.low
    high_bound = env.observation_space.high
    env_shape = (high_bound[0] - low_bound[0] + 1, high_bound[1] - low_bound[1] + 1)
    episode_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        state = int(np.ravel_multi_index(state,env_shape))
        done = False
        absorbing = False
        episode_reward = 0
        while not (done or absorbing):
            state, reward, done, absorbing = agent.play(state)
            episode_reward += reward
        episode_rewards.append(episode_reward)
        if episode % 10 == 0:
            print(f'Episode: {episode}, Reward: {episode_reward}, Epsilon: {agent.eps}')
    return episode_rewards


env = mo_gym.make("deep-sea-treasure-v0")
agent = Agent(env)
rewards = train(agent, 5000)
