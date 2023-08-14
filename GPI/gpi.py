from typing import Union, Callable
import numpy as np
import torch as th
import wandb
from copy import deepcopy
from utils.utils import eval_mo
from visualization import Visualization
from utils_log import get_eval_w, calc_opt_reward, linear, tensor_to_numpy, calc_hypervolume, pareto_filter
from neptune_logger import NeptuneLogger
from rl_algorithm import RLAlgorithm
import pandas as pd


class GPI(RLAlgorithm):

    def __init__(self,
                 env,
                 algorithm_constructor: Callable,
                 log: bool = True,
                 logger: NeptuneLogger = None,
                 project_name: str = 'gpi',
                 experiment_name: str = 'gpi',
                 device: Union[th.device, str] = 'auto'):
        super(GPI, self).__init__(env, device)

        self.algorithm_constructor = algorithm_constructor
        self.policies = []
        self.tasks = []

        self.df_results = pd.DataFrame()
        self.reward_dim = 2
        self.n_obj = 2
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
        
        self.global_step = 0
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

        self.log = log
        if self.log:
            self.nl = logger
            # self.setup_wandb(project_name, experiment_name)

    def eval(self, obs, w, return_policy_index=False, exclude=None) -> int:
        if not hasattr(self.policies[0], 'q_table'):
            if isinstance(obs, np.ndarray):
                obs = th.tensor(obs).float().to(self.device)
                w = th.tensor(w).float().to(self.device)
            q_vals = th.stack([policy.q_values(obs, w) for policy in self.policies])
            max_q, a = th.max(q_vals, dim=2)
            policy_index = th.argmax(max_q)
            if return_policy_index:
                return a[policy_index].detach().long().item(), policy_index.item()
            return a[policy_index].detach().long().item()
        else:
            q_vals = np.stack([policy.q_values(obs, w) for policy in self.policies if policy is not exclude])
            policy_index, action = np.unravel_index(np.argmax(q_vals), q_vals.shape)
            if return_policy_index:
                return action, policy_index
            return action
     
    def max_q(self, obs, w, tensor=False, exclude=None):
        if tensor:
            with th.no_grad():
                psi_values = th.stack([policy.target_psi_net(obs) for policy in self.policies if policy is not exclude])
                q_values = th.einsum('r,psar->psa', w, psi_values)
                max_q, a = th.max(q_values, dim=2)
                polices = th.argmax(max_q, dim=0)
                max_acts = a.gather(0, polices.unsqueeze(0)).squeeze(0)
                psi_i = psi_values.gather(0, polices.reshape(1,-1,1,1).expand(1, psi_values.size(1), psi_values.size(2), psi_values.size(3))).squeeze(0)
                max_psis = psi_i.gather(1, max_acts.reshape(-1,1,1).expand(psi_i.size(0), 1, psi_i.size(2))).squeeze(1)
                return max_psis
        else:
            q_vals = np.stack([policy.q_values(obs, w) for policy in self.policies])
            policy_ind, action = np.unravel_index(np.argmax(q_vals), q_vals.shape)
            return self.policies[policy_ind].q_table[tuple(obs)][action]
    
    def delete_policies(self, delete_indx):
        for i in sorted(delete_indx, reverse=True):
            self.policies.pop(i)
            self.tasks.pop(i)

    def learn(self, w, total_timesteps, total_episodes=None, reset_num_timesteps=False, eval_env=None, eval_freq=1000, use_gpi=True, reset_learning_starts=True, new_policy=True, reuse_value_ind=None):
        if new_policy:
            new_policy = self.algorithm_constructor()
            self.policies.append(new_policy)
        self.tasks.append(w)
        
        self.policies[-1].gpi = self if use_gpi else None

        if self.log:
            self.policies[-1].log = self.log
            #self.policies[-1].writer = self.writer
            # wandb.config.update(self.policies[-1].get_config())

        if len(self.policies) > 1:
            self.policies[-1].num_timesteps = self.policies[-2].num_timesteps
            self.policies[-1].num_episodes = self.policies[-2].num_episodes
            if reset_learning_starts:
                self.policies[-1].learning_starts = self.policies[-2].num_timesteps  # to reset exploration schedule

            if reuse_value_ind is not None:
                if hasattr(self.policies[-1], 'q_table'):
                    self.policies[-1].q_table = deepcopy(self.policies[reuse_value_ind].q_table)
                else:
                    self.policies[-1].psi_net.load_state_dict(self.policies[reuse_value_ind].psi_net.state_dict())
                    self.policies[-1].target_psi_net.load_state_dict(self.policies[reuse_value_ind].psi_net.state_dict())

            self.policies[-1].replay_buffer = self.policies[-2].replay_buffer

        self.policies[-1].learn(w=w,
                                total_timesteps=total_timesteps, 
                                total_episodes=total_episodes,
                                reset_num_timesteps=reset_num_timesteps,
                                eval_env=eval_env,
                                eval_freq=eval_freq)

    @property
    def gamma(self):
        return self.policies[0].gamma

    def train(self):
        pass

    def get_config(self) -> dict:
        if len(self.policies) > 0:
            return self.policies[0].get_config()
        return {}
    def _log_figures(self, step, e_reutrns,  last=False):
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

    def get_metric(self, global_step):
        self.df_results = pd.DataFrame(columns=self.pref_cols + self.reward_cols, index=np.arange(len(self.eval_prefs)))
        e_returns = []
        actions = []
        for n, w in enumerate(self.eval_prefs):
            _,_,reward,_,action =  eval_mo(self, self.env, w)
            actions.append(action)
            e_returns.append(reward)
            total_utility = self.u_func(np.array(reward,dtype=float), np.array(self.eval_prefs[n],dtype=float))
            self.df_results.loc[n, f'utility'] = total_utility
            for obj in range(self.n_obj):
                    self.df_results.loc[n, f'pref_{obj}'] = self.eval_prefs[n][obj]
                    self.df_results.loc[n, f'reward_{obj}'] = reward[obj]
        self.actions = [item for sublist in actions for item in sublist]
        self.calc_metrics()
        self.nl.log_metric(metric_value=self.pref_loss, metric_name='pref_loss_0', mode='metrics', step=global_step)
        self.nl.log_metric(metric_value=self.utility, metric_name='utility_0', mode='metrics', step=global_step)
        self.nl.log_metric(metric_value=self.hypervolume, metric_name='hypervolume_0', mode='metrics',
                        step=global_step)
        self.nl.log_metric(metric_value=self.total_rewards.mean(), metric_name='total_reward',
                        mode='metrics', step=global_step)
        self._log_figures(global_step, np.array(e_returns))
        self._log_results(global_step, self.df_results)

        return 

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

    def get_frontier(self, empiric=False):
        if empiric or self.front is None:
            return self.empiric_frontier
        else:
            return self.front
        
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
