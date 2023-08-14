"""Pareto Conditioned Network. Code adapted from https://github.com/mathieu-reymond/pareto-conditioned-networks ."""
import heapq
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from visualization import Visualization
from utils import get_eval_w, calc_opt_reward, calc_hypervolume, pareto_filter, linear, tensor_to_numpy
from neptune_logger import NeptuneLogger

from morl_algorithm import MOAgent, MOPolicy
from pareto import get_non_dominated_inds
from performance_indicators import hypervolume
from utils import log_all_multi_policy_metrics

tags = ["pcn","ser","deep-sea-treasure",'dst_pcn_ser_test']

def crowding_distance(points):
    """Compute the crowding distance of a set of points."""
    # first normalize across dimensions
    points = (points - points.min(axis=0)) / (points.ptp(axis=0) + 1e-8)
    # sort points per dimension
    dim_sorted = np.argsort(points, axis=0)
    point_sorted = np.take_along_axis(points, dim_sorted, axis=0)
    # compute distances between lower and higher point
    distances = np.abs(point_sorted[:-2] - point_sorted[2:])
    # pad extrema's with 1, for each dimension
    distances = np.pad(distances, ((1,), (0,)), constant_values=1)
    # sum distances of each dimension of the same point
    crowding = np.zeros(points.shape)
    crowding[dim_sorted, np.arange(points.shape[-1])] = distances
    crowding = np.sum(crowding, axis=-1)
    return crowding


@dataclass
class Transition:
    """Transition dataclass."""

    observation: np.ndarray
    action: int
    reward: np.ndarray
    next_observation: np.ndarray
    terminal: bool


class Model(nn.Module):
    """Model for the PCN."""

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray, hidden_dim: int = 64):
        """Initialize the PCN model."""
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.scaling_factor = nn.Parameter(th.tensor(scaling_factor).float(), requires_grad=False)
        self.hidden_dim = hidden_dim

        self.s_emb = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(self.reward_dim + 1, self.hidden_dim), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.LogSoftmax(1),
        )

    def forward(self, state, desired_return, desired_horizon):
        """Return log-probabilities of actions."""
        c = th.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c * self.scaling_factor
        s = self.s_emb(state.float())
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        log_prob = self.fc(s * c)
        return log_prob


class PCN(MOAgent, MOPolicy):
    """Pareto Conditioned Networks (PCN).

    Reymond, M., Bargiacchi, E., & Nowé, A. (2022, May). Pareto Conditioned Networks.
    In Proceedings of the 21st International Conference on Autonomous Agents
    and Multiagent Systems (pp. 1110-1118).
    https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1110.pdf

    ## Credits

    This code is a refactor of the code from the authors of the paper, available at:
    https://github.com/mathieu-reymond/pareto-conditioned-networks
    """

    def __init__(
        self,
        env: Optional[gym.Env],
        scaling_factor: np.ndarray,
        learning_rate: float = 1e-2,
        gamma: float = 1.0,
        batch_size: int = 32,
        hidden_dim: int = 64,
        project_name: str = "MORL-Baselines",
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """Initialize PCN agent.

        Args:
            env (Optional[gym.Env]): Gym environment.
            scaling_factor (np.ndarray): Scaling factor for the desired return and horizon used in the model.
            learning_rate (float, optional): Learning rate. Defaults to 1e-2.
            gamma (float, optional): Discount factor. Defaults to 1.0.
            batch_size (int, optional): Batch size. Defaults to 32.
            hidden_dim (int, optional): Hidden dimension. Defaults to 64.
            project_name (str, optional): Name of the project for wandb. Defaults to "MORL-Baselines".
            experiment_name (str, optional): Name of the experiment for wandb. Defaults to "PCN".
            wandb_entity (Optional[str], optional): Entity for wandb. Defaults to None.
            log (bool, optional): Whether to log to wandb. Defaults to True.
            seed (Optional[int], optional): Seed for reproducibility. Defaults to None.
            device (Union[th.device, str], optional): Device to use. Defaults to "auto".
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device)
        self.experience_replay = []  # List of (distance, time_step, transition)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.scaling_factor = scaling_factor
        self.desired_return = None
        self.desired_horizon = None

        self.model = Model(
            self.observation_dim, self.action_dim, self.reward_dim, self.scaling_factor, hidden_dim=self.hidden_dim
        ).to(self.device)
        self.opt = th.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        
        self.log = log
        self.project_name = project_name
        self.nl = NeptuneLogger()
        ## for evaluation
        self.df_results = pd.DataFrame()
        self.n_obj = self.reward_dim
        self.actions = []
        self.pareto_idx = []
        self.pareto_results = []
        self.empiric_frontier = np.array([])
        self.df_empiric_frontier = pd.DataFrame(self.empiric_frontier)
        self.pref_cols = []
        self.reward_cols = []
        for obj in range(self.n_obj):
            self.pref_cols.append(f'pref_{obj}')
            self.reward_cols.append(f'reward_{obj}')
        
        self.total_rewards = 0
        self.pref_loss = 0
        self.utility = 0
        self.hypervolume = 0
        self.utopia = np.array([26. ,  0.8])
        self.dystopia = np.array([-10.8, -28. ])
        self.total_rewards = 0
        self.empiric_frontier =  np.array([])
        self.n = 32
        self.eval_w = get_eval_w(self.n, self.reward_dim)
        self.eval_prefs = self.eval_w
        self.front = np.array([[0.7, -1],[8.2,-3],[11.5,-5],[14,-7],[15.1,-8],[16.1,-9],[19.6,-13],[20.3,-14],[22.4,-17],[23.7,-19]])
        self.w_front = calc_opt_reward(self.eval_w, self.front, u_func=None)
        self.u_func = lambda x, y: tensor_to_numpy(linear(th.tensor(x, device="cpu"),
                                                       th.tensor(y, device="cpu")))
        self.vis = Visualization(self.reward_dim, self.front)
        if log:
            self.nl.start(run_name=self.project_name, tags=tags)

    def get_config(self) -> dict:
        """Get configuration of PCN model."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "scaling_factor": self.scaling_factor,
            "seed": self.seed,
        }

    def update(self):
        """Update PCN model."""
        batch = []
        # randomly choose episodes from experience buffer
        s_i = self.np_random.choice(np.arange(len(self.experience_replay)), size=self.batch_size, replace=True)
        for i in s_i:
            # episode is tuple (return, transitions)
            ep = self.experience_replay[i][2]
            # choose random timestep from episode,
            # use it's return and leftover timesteps as desired return and horizon
            t = self.np_random.integers(0, len(ep))
            # reward contains return until end of episode
            s_t, a_t, r_t, h_t = ep[t].observation, ep[t].action, np.float32(ep[t].reward), np.float32(len(ep) - t)
            batch.append((s_t, a_t, r_t, h_t))

        obs, actions, desired_return, desired_horizon = zip(*batch)
        log_prob = self.model(
            th.tensor(obs).to(self.device),
            th.tensor(desired_return).to(self.device),
            th.tensor(desired_horizon).unsqueeze(1).to(self.device),
        )

        self.opt.zero_grad()
        # one-hot of action for CE loss
        actions = F.one_hot(th.tensor(actions).long().to(self.device), len(log_prob[0]))
        # cross-entropy loss
        l = th.sum(-actions * log_prob, -1)
        l = l.mean()
        l.backward()
        self.opt.step()

        return l.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def _add_episode(self, transitions: List[Transition], max_size: int, step: int) -> None:
        # compute return
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += self.gamma * transitions[i + 1].reward
        # pop smallest episode of heap if full, add new episode
        # heap is sorted by negative distance, (updated in nlargest)
        # put positive number to ensure that new item stays in the heap
        if len(self.experience_replay) == max_size:
            heapq.heappushpop(self.experience_replay, (1, step, transitions))
        else:
            heapq.heappush(self.experience_replay, (1, step, transitions))

    def _nlargest(self, n, threshold=0.2):
        """See Section 4.4 of https://arxiv.org/pdf/2204.05036.pdf for details."""
        returns = np.array([e[2][0].reward for e in self.experience_replay])
        # crowding distance of each point, check ones that are too close together
        distances = crowding_distance(returns)
        sma = np.argwhere(distances <= threshold).flatten()

        non_dominated_i = get_non_dominated_inds(returns)
        non_dominated = returns[non_dominated_i]
        # we will compute distance of each point with each non-dominated point,
        # duplicate each point with number of non_dominated to compute respective distance
        returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))
        # distance to closest non_dominated point
        l2 = np.min(np.linalg.norm(returns_exp - non_dominated, axis=-1), axis=-1) * -1
        # all points that are too close together (crowding distance < threshold) get a penalty
        non_dominated_i = np.nonzero(non_dominated_i)[0]
        _, unique_i = np.unique(non_dominated, axis=0, return_index=True)
        unique_i = non_dominated_i[unique_i]
        duplicates = np.ones(len(l2), dtype=bool)
        duplicates[unique_i] = False
        l2[duplicates] -= 1e-5
        l2[sma] *= 2

        sorted_i = np.argsort(l2)
        largest = [self.experience_replay[i] for i in sorted_i[-n:]]
        # before returning largest elements, update all distances in heap
        for i in range(len(l2)):
            self.experience_replay[i] = (l2[i], self.experience_replay[i][1], self.experience_replay[i][2])
        heapq.heapify(self.experience_replay)
        return largest

    def _choose_commands(self, num_episodes: int):
        # get best episodes, according to their crowding distance
        episodes = self._nlargest(num_episodes)
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        # keep only non-dominated returns
        nd_i = get_non_dominated_inds(np.array(returns))
        returns = np.array(returns)[nd_i]
        horizons = np.array(horizons)[nd_i]
        # pick random return from random best episode
        r_i = self.np_random.integers(0, len(returns))
        desired_horizon = np.float32(horizons[r_i] - 2)
        # mean and std per objective
        _, s = np.mean(returns, axis=0), np.std(returns, axis=0)
        # desired return is sampled from [M, M+S], to try to do better than mean return
        desired_return = returns[r_i].copy()
        # random objective
        r_i = self.np_random.integers(0, len(desired_return))
        desired_return[r_i] += self.np_random.uniform(high=s[r_i])
        desired_return = np.float32(desired_return)
        return desired_return, desired_horizon

    def _act(self, obs: np.ndarray, desired_return, desired_horizon) -> int:
        log_probs = self.model(
            th.tensor([obs]).float().to(self.device),
            th.tensor([desired_return]).float().to(self.device),
            th.tensor([desired_horizon]).unsqueeze(1).float().to(self.device),
        )
        log_probs = log_probs.detach().cpu().numpy()[0]
        action = self.np_random.choice(np.arange(len(log_probs)), p=np.exp(log_probs))
        return action

    def _run_episode(self, env, desired_return, desired_horizon, max_return):
        transitions = []
        obs, _ = env.reset()
        done = False
        while not done:
            action = self._act(obs, desired_return, desired_horizon)
            n_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            transitions.append(
                Transition(
                    observation=obs,
                    action=action,
                    reward=np.float32(reward).copy(),
                    next_observation=n_obs,
                    terminal=terminated,
                )
            )

            obs = n_obs
            # clip desired return, to return-upper-bound,
            # to avoid negative returns giving impossible desired returns
            desired_return = np.clip(desired_return - reward, None, max_return, dtype=np.float32)
            # clip desired horizon to avoid negative horizons
            desired_horizon = np.float32(max(desired_horizon - 1, 1.0))
        return transitions

    def set_desired_return_and_horizon(self, desired_return: np.ndarray, desired_horizon: int):
        """Set desired return and horizon for evaluation."""
        self.desired_return = desired_return
        self.desired_horizon = desired_horizon

    def eval(self, obs, w=None):
        """Evaluate policy action for a given observation."""
        return self._act(obs, self.desired_return, self.desired_horizon)
    
    def get_wfront(self,n=32):
        eval_w = get_eval_w(n, 2)
        rewards = self.front
    
        rewards_1 = rewards[:,0]
        rewards_2 = []
        for i in range(1, int(((rewards[:,1])*-1).max())+1):
            rewards_2.append(i)
        rewards_2=np.array(rewards_2)
        t_func = lambda x, y: (x / y)

        u_func = lambda x, y: abs((x - y))

        nf1 = t_func(rewards_1, (rewards_1.max(axis=0)-rewards_1.min(axis=0)))
        nf2 = t_func(rewards_2, (rewards_2.max(axis=0)-rewards_2.min(axis=0)))
        nf1 = np.float32(nf1)
        nf2 = np.float32(nf2)
        returns = []
        horizons = []
        for n , w in enumerate(eval_w):
            id_1 = u_func(nf1 , w[0]).argmin()
            id_2 = u_func(nf2 , w[1]).argmin()
            returns.append([rewards_1[id_1], rewards_2[id_2]-20])
            horizons.append(abs(rewards_2[id_2]-20))
        return returns, horizons
    
    def get_wfront2(self):
        w_front = self.w_front
        return w_front, w_front[:,1]*-1

    def evaluate(self, env, max_return, n=10, wfront=''):
        """Evaluate policy in the given environment."""
        self.df_results = pd.DataFrame(columns=self.pref_cols + self.reward_cols, index=np.arange(len(self.eval_prefs)))
        if wfront == '':
            episodes = self._nlargest(n)
            returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
            returns = np.float32(returns)
            horizons = np.float32(horizons)
        elif wfront == 'wfront':
            returns, horizons = self.get_wfront(n)
            returns = np.float32(returns)
            horizons = np.float32(horizons)
            n+=1
        elif wfront == 'wfront2':
            returns, horizons = self.get_wfront2()
            n+=1
        e_returns = []
        num = n
        self.actions = []
        for n in range(num):
            transitions = self._run_episode(env, returns[n], np.float32(horizons[n] - 2), max_return)
            # compute return
            for i in reversed(range(len(transitions) - 1)):
                transitions[i].reward += self.gamma * transitions[i + 1].reward
            e_returns.append(transitions[0].reward)
            self.actions.append(transitions[0].action)

            total_utility = self.u_func(e_returns[n], self.eval_w[n])
            self.df_results.loc[n, f'utility'] = total_utility
            for obj in range(self.n_obj):
                    self.df_results.loc[n, f'pref_{obj}'] = self.eval_prefs[n][obj]
                    self.df_results.loc[n, f'reward_{obj}'] = e_returns[n][obj]
        

        distances = np.linalg.norm(np.array(returns) - np.array(e_returns), axis=-1)
        # df = [[returns[x],horizons[x],e_returns[x]] for x in range(len(e_returns))]
        # df = pd.DataFrame(df,columns=['desired_return', 'horizon', 'eval_return'])
        # df.to_csv(f'PCN/results3/pcn_wfront_{self.global_step}.csv')
        return e_returns, np.array(returns), distances

    def save(self, filename: str = "PCN_model", savedir: str = "weights"):
        """Save PCN."""
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        th.save(self.model, f"{savedir}/{filename}.pt")

    def _log_figures(self, step,e_reutrns,  last=False):
                    log_fun = self.nl.log_fig if not last else self.nl.upload_fig
                    fig1 = self.vis.gen_hist(self.actions, step, 'Action_dist')
                    log_fun(fig1, f'action_distribution_ul', 'plots')
                    if self.reward_dim < 4:
                        fig2 = self.vis.gen_plt(self.utopia, self.dystopia, self.utopia,
                                                self.dystopia, self.total_rewards,
                                                self.pareto_results, step, self.get_frontier())
                        log_fun(fig2, f'pareto_ul', 'plots')
                    if self.reward_dim == 2:
                        fig2 = self.vis.gen_xy_plot(self.eval_w, e_reutrns, step,
                                                    'Rewards_Pref', self.w_front)
                        log_fun(fig2, f'pareto_xy_ul', 'plots')

    def _log_results(self, step, df_results):
        self.nl.log_dataframe(df=df_results, df_name=f'results_{step}', mode='csvs')
        self.nl.log_dataframe(df=self.df_empiric_frontier, df_name=f'empiric_frontier_{step}', mode='csvs')

    def get_frontier(self, empiric=False):
        if empiric or self.front is None:
            return self.empiric_frontier
        else:
            return self.front

    def calc_metrics(self):
        """

        """
        
        self.eval_prefs = self.df_results[self.pref_cols].values
        self.total_rewards = self.df_results[self.reward_cols].values.astype(float)
        self.pareto_results, self.pareto_idx = pareto_filter(self.total_rewards)
        pareto_frontier = self.get_frontier()
        if len(pareto_frontier) > 0:
            self.reward_opt = calc_opt_reward(self.eval_prefs, pareto_frontier)
            loss_opt = abs(self.total_rewards - self.reward_opt)
            self.pref_loss = np.sqrt((loss_opt ** 2).sum(axis=1)).mean()

        self.utility = (self.eval_prefs * self.total_rewards).sum(axis=1).mean()
        self.hypervolume = calc_hypervolume(self.total_rewards, self.utopia, self.dystopia)
        self.update_empiric_frontier(self.pareto_results)

    def update_empiric_frontier(self, pareto_results):
        """

        Parameters
        ----------
        pareto_results
        """
        if len(self.empiric_frontier) == 0:
            self.empiric_frontier = pareto_results
        else:
            self.empiric_frontier = np.concatenate([self.empiric_frontier, pareto_results])
        self.empiric_frontier, _ = pareto_filter(self.empiric_frontier)
        self.df_empiric_frontier = pd.DataFrame(self.empiric_frontier)

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_er_episodes: int = 500,
        num_step_episodes: int = 10,
        num_model_updates: int = 100,
        max_return: np.ndarray = 250.0,
        max_buffer_size: int = 500,
    ):
        """Train PCN.

        Args:
            total_timesteps: total number of time steps to train for
            eval_env: environment for evaluation
            ref_point: reference point for hypervolume calculation
            known_pareto_front: Optimal pareto front for metrics calculation, if known.
            num_er_episodes: number of episodes to fill experience replay buffer
            num_step_episodes: number of steps per episode
            num_model_updates: number of model updates per episode
            max_return: maximum return for clipping desired return
            max_buffer_size: maximum buffer size
        """
        # if self.log:
        #     self.register_additional_config({"ref_point": ref_point.tolist(), "known_front": known_pareto_front})
        self.global_step = 0
        total_episodes = num_er_episodes
        n_checkpoints = 0

        # fill buffer with random episodes
        self.experience_replay = []
        for _ in range(num_er_episodes):
            transitions = []
            obs, _ = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                n_obs, reward, terminated, truncated, _ = self.env.step(action)
                transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, terminated))
                done = terminated or truncated
                obs = n_obs
                self.global_step += 1
            # add episode in-place
            self._add_episode(transitions, max_size=max_buffer_size, step=self.global_step)

        while self.global_step < total_timesteps:
            loss = []
            entropy = []
            for _ in range(num_model_updates):
                l, lp = self.update()
                loss.append(l)
                ent = np.sum(-np.exp(lp) * lp)
                entropy.append(ent)

            desired_return, desired_horizon = self._choose_commands(num_er_episodes)

            # get all leaves, contain biggest elements, experience_replay got heapified in choose_commands
            leaves_r = np.array([e[2][0].reward for e in self.experience_replay[len(self.experience_replay) // 2 :]])
            # leaves_h = np.array([len(e[2]) for e in self.experience_replay[len(self.experience_replay) // 2 :]])

            if self.log:
                hv = hypervolume(ref_point, leaves_r)
                hv_est = hv
                self.nl.log_metric(metric_name="hypervolume_0",metric_value=hv_est,step=self.global_step,mode='metrics')
                self.nl.log_metric(metric_name="loss",metric_value=np.mean(loss),step=self.global_step,mode='metrics')
                self.nl.log_metric(metric_name="entropy",metric_value=np.mean(entropy),step=self.global_step,mode='metrics')

            returns = []
            horizons = []
            for _ in range(num_step_episodes):
                transitions = self._run_episode(self.env, desired_return, desired_horizon, max_return)
                self.global_step += len(transitions)
                self._add_episode(transitions, max_size=max_buffer_size, step=self.global_step)
                returns.append(transitions[0].reward)
                horizons.append(len(transitions))

            total_episodes += num_step_episodes
            if self.log:
                self.nl.log_metric(metric_name="total_episodes",metric_value=total_episodes,step=self.global_step,mode='metrics')
                self.nl.log_metric(metric_name="horizon_desired",metric_value=desired_horizon,step=self.global_step,mode='metrics')
                self.nl.log_metric(metric_name="mean_horizon_distance",metric_value=np.linalg.norm(np.mean(horizons) - desired_horizon),step=self.global_step,mode='metrics')

                for i in range(self.reward_dim):
                    self.nl.log_metric(metric_name=f"desired_return_{i}",metric_value=desired_return[i],step=self.global_step,mode='metrics')
                    self.nl.log_metric(metric_name=f"mean_return_{i}",metric_value=np.mean(np.array(returns)[:, i]),step=self.global_step,mode='metrics')
                    self.nl.log_metric(metric_name=f"mean_return_distance_{i}",metric_value=np.linalg.norm(np.mean(np.array(returns)[:, i]) - desired_return[i]),step=self.global_step,mode='metrics')

            print(
                f"step {self.global_step} \t return {np.mean(returns, axis=0)}, ({np.std(returns, axis=0)}) \t loss {np.mean(loss):.3E}"
            )

            if self.global_step >= (n_checkpoints + 1) * total_timesteps / 100:
                self.save()
                n_checkpoints += 1
                n_points = 32
                wfront='wfront2'
                
                e_returns, _, _ = self.evaluate(eval_env, max_return, n=n_points,wfront = wfront)
                # df = [[self.eval_w[x][0],self.eval_w[x][1],e_returns[x][0],e_returns[x][1]] for x in range(len(e_returns))]
                # df = pd.DataFrame(df, columns=["pref_0","pref_1", "reward_0", "reward_1"])
                self.calc_metrics()
                
                if self.log:
                    self.nl.log_metric(metric_value=self.pref_loss, metric_name='pref_loss', mode='metrics', step=self.global_step)
                    self.nl.log_metric(metric_value=self.utility, metric_name='utility', mode='metrics', step=self.global_step)
                    self.nl.log_metric(metric_value=self.hypervolume, metric_name='hypervolume', mode='metrics',
                                    step=self.global_step)
                    self.nl.log_metric(metric_value=self.total_rewards.mean(), metric_name='total_reward',
                                    mode='metrics', step=self.global_step)
                    self._log_figures(self.global_step, np.array(e_returns))
                    self._log_results(self.global_step, self.df_results)
                
                if self.log:
                    self.nl.log_metric(metric_name="global_step",metric_value=self.global_step,step=self.global_step,mode='metrics')
                    log_all_multi_policy_metrics(self.nl,
                        current_front=e_returns,
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        ref_front=known_pareto_front,
                    )