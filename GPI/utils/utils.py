import os
import numpy as np
import torch as th
import random
from torch import nn
from typing import Iterable, List, Union
from pymoo.factory import get_performance_indicator
import pandas as pd

def layer_init(layer, method='xavier', weight_gain=1, bias_const=0):
    if isinstance(layer, nn.Linear):
        if method == "xavier":
            th.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif method == "orthogonal":
            th.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        th.nn.init.constant_(layer.bias, bias_const)

def polyak_update(params: Iterable[th.nn.Parameter], target_params: Iterable[th.nn.Parameter], tau: float) -> None:
    with th.no_grad():
        for param, target_param in zip(params, target_params):
            if tau == 1:
                target_param.data.copy_(param.data)
            else:
                target_param.data.mul_(1.0 - tau)
                th.add(target_param.data, param.data, alpha=tau, out=target_param.data)

def huber(x, min_priority=0.01):
    return th.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).mean()

def generate_weights(count=1, n=3, m=1):
    """Source: https://github.com/axelabels/DynMORL/blob/db15c29bc2cf149c9bda6b8890fee05b1ac1e19e/utils.py#L281"""
    all_weights = []

    target = np.random.dirichlet(np.ones(n), 1)[0]
    prev_t = target
    for _ in range(count // m):
        target = np.random.dirichlet(np.ones(n), 1)[0]
        if m == 1:
            all_weights.append(target)
        else:
            for i in range(m):
                i_w = target * (i + 1) / float(m) + prev_t * (m - i - 1) / float(m)
                all_weights.append(i_w)
        prev_t = target + 0.

    return all_weights

def random_weights(dim, seed=None, n=1):
    """ Generate random normalized weights from a Dirichlet distribution alpha=1
        Args:
            dim: size of the weight vector
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random
    weights = []
    for _ in range(n):
        w = rng.dirichlet(np.ones(dim))
        weights.append(w)
    if n == 1:
        return weights[0]
    return weights

def eval(agent, env, render=False):
    obs = env.reset()
    done = False
    total_reward, discounted_return = 0.0, 0.0
    gamma = 1.0
    while not done:
        if render:
            env.render()
        obs, r, done, info = env.step(agent.eval(obs))
        total_reward += r
        discounted_return += gamma * r
        gamma *= agent.gamma
    return total_reward, discounted_return

def eval_mo(agent, env, w, render=False):
    """
    Returns:
        w.total_reward, w.return, total vec reward, vec return
    """
    obs = env.reset()
    done = False
    total_vec_reward, vec_return = np.zeros_like(w), np.zeros_like(w)
    gamma = 1.0
    actions = []
    while not done:
        if render:
            env.render()
        action = agent.eval(obs, w)
        obs, r, done, info = env.step(action)
        total_vec_reward += info['phi']
        vec_return += gamma * info['phi']
        gamma *= agent.gamma
        actions.append(action)
    return np.dot(w, total_vec_reward), np.dot(w,vec_return), total_vec_reward, vec_return, actions

def policy_evaluation_mo(agent, env, w, rep=10, return_scalarized_value=False):
    """Returns vectorized value of the policy (mean of the returns)"""
    if return_scalarized_value:
        returns = [eval_mo(agent, env, w)[1] for _ in range(rep)]
    else:
        returns = [eval_mo(agent, env, w)[3] for _ in range(rep)]
    return np.mean(returns, axis=0)

def eval_test_tasks(agent, env, tasks, rep=10):
    """Returns mean scalar value of the policy"""
    returns = [policy_evaluation_mo(agent, env, w, rep=rep, return_scalarized_value=True) for w in tasks]
    return np.mean(returns, axis=0)

def moving_average(interval: Union[np.array,List], window_size: int) -> np.array:
    if window_size == 1:
        return interval
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def linearly_decaying_epsilon(initial_epsilon, decay_period, step, warmup_steps, final_epsilon):
    """Returns the current epsilon for the agent's epsilon-greedy policy.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.
    Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.
    Returns:
    A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (initial_epsilon - final_epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - final_epsilon)
    return final_epsilon + bonus

def hypervolume(ref_point: np.ndarray, points: List[np.ndarray]) -> float:
    hv = get_performance_indicator("hv", ref_point=ref_point*-1)
    return hv.do(np.array(points)*-1)

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = True

def get_eval_w(n, dim):
    """

    Parameters
    ----------
    n
    dim

    Returns
    -------

    """
    step = 1 / n
    b = []
    if dim == 4:
        for b1 in np.arange(0, 1 + .00000001, step):
            b234 = 1 - b1
            for b2 in np.arange(0, b234 + .00000001, step):
                b34 = b234 - b2
                for b3 in np.arange(0, b34 + .00000001, step):
                    b4 = b34 - b3
                    b += [[b1, b2, b3, abs(b4)]]
    elif dim == 5:
        for b0 in np.arange(0, 1 + .00000001, step):
            b2345 = 1 - b0
            for b1 in np.arange(0, b2345 + .00000001, step):
                b234 = b2345 - b1
                for b2 in np.arange(0, b234 + .00000001, step):
                    b34 = b234 - b2
                    for b3 in np.arange(0, b34 + .00000001, step):
                        b4 = b34 - b3
                        b += [[b0, b1, b2, b3, abs(b4)]]
    elif dim == 6:
        for b1m in np.arange(0, 1 + .00000001, step):
            b23456 = 1 - b1m
            for b0 in np.arange(0, b23456 + .00000001, step):
                b2345 = b23456 - b0
                for b1 in np.arange(0, b2345 + .00000001, step):
                    b234 = b2345 - b1
                    for b2 in np.arange(0, b234 + .00000001, step):
                        b34 = b234 - b2
                        for b3 in np.arange(0, b34 + .00000001, step):
                            b4 = b34 - b3
                            b += [[b1m, b0, b1, b2, b3, abs(b4)]]
    elif dim == 3:
        for b1 in np.arange(0, 1 + .00000001, step):
            b234 = 1 - b1
            for b2 in np.arange(0, b234 + .00000001, step):
                b3 = b234 - b2
                b += [[b1, b2, abs(b3)]]
    elif dim == 2:
        for b1 in np.arange(0, 1 + .00000001, step):
            b2 = 1 - b1
            b += [[b1, abs(b2)]]
    elif dim == 1:
        b = [[1]]
    return np.array(b)

if __name__ == '__main__':

    #print(generate_weights(32,2,10))
    #print(random_weights(2))
    eee = []
    returns = [np.array([ 0.7, -1. ]), np.array([ 0.7, -1. ]), np.array([ 0.7, -1. ]), np.array([ 0.7, -1. ]), np.array([ 0.7, -1. ]), np.array([ 0.7, -1. ]), np.array([ 0.7, -1. ]), np.array([ 8.037, -2.97 ]), np.array([ 8.037, -2.97 ]), np.array([ 8.037, -2.97 ]), np.array([ 8.037, -2.97 ]), np.array([ 8.037, -2.97 ]), np.array([ 8.037, -2.97 ]), np.array([11.047, -4.901]), np.array([11.047, -4.901]), np.array([11.047, -4.901]), np.array([13.181, -6.793]), np.array([14.074, -7.726]), np.array([14.856, -8.648]), np.array([ 17.373, -12.248]), np.array([ 17.373, -12.248]), np.array([ 17.373, -12.248]), np.array([ 19.073, -15.706]), np.array([ 19.778, -17.383]), np.array([ 19.778, -17.383]), np.array([ 19.778, -17.383]), np.array([ 19.778, -17.383]), np.array([ 19.778, -17.383]), np.array([ 19.778, -17.383]), np.array([ 19.778, -17.383]), np.array([ 19.778, -17.383]), np.array([ 19.778, -17.383]), np.array([ 19.778, -17.383])]
    w = get_eval_w(32, 2)
    for i in range(len(returns)):
        eee.append({'w': w[i], 'returns': returns[i]})
    df = pd.DataFrame(eee)
    df.to_csv('returns.csv')
