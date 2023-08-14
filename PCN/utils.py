"""General utils for the MORL baselines."""
import os
import random
from functools import lru_cache
from typing import Iterable, List, Optional
from torch.distributions import uniform
import torch.nn as nn
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import torch as th
from pymoo.util.ref_dirs import get_reference_directions
from torch.utils.tensorboard import SummaryWriter

from performance_indicators import (
    expected_utility,
    hypervolume,
    igd,
    maximum_utility_loss,
    sparsity,
)

TORCH_PRECISION = torch.float32
TORCH_PRECISION_HIGH = torch.float64
USE_CUDA = True

@th.no_grad()
def layer_init(layer, method="orthogonal", weight_gain: float = 1, bias_const: float = 0) -> None:
    """Initialize a layer with the given method.

    Args:
        layer: The layer to initialize.
        method: The initialization method to use.
        weight_gain: The gain for the weights.
        bias_const: The constant for the bias.
    """
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if method == "xavier":
            th.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif method == "orthogonal":
            th.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        th.nn.init.constant_(layer.bias, bias_const)


@th.no_grad()
def polyak_update(
    params: Iterable[th.nn.Parameter],
    target_params: Iterable[th.nn.Parameter],
    tau: float,
) -> None:
    """Polyak averaging for target network parameters.

    Args:
        params: The parameters to update.
        target_params: The target parameters.
        tau: The polyak averaging coefficient (usually small).

    """
    for param, target_param in zip(params, target_params):
        if tau == 1:
            target_param.data.copy_(param.data)
        else:
            target_param.data.mul_(1.0 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def get_grad_norm(params: Iterable[th.nn.Parameter]) -> th.Tensor:
    """This is how the grad norm is computed inside torch.nn.clip_grad_norm_().

    Args:
        params: The parameters to compute the grad norm for.

    Returns:
        The grad norm.
    """
    parameters = [p for p in params if p.grad is not None]
    if len(parameters) == 0:
        return th.tensor(0.0)
    device = parameters[0].grad.device
    total_norm = th.norm(th.stack([th.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
    return total_norm


def huber(x, min_priority=0.01):
    """Huber loss function.

    Args:
        x: The input tensor.
        min_priority: The minimum priority.

    Returns:
        The huber loss.
    """
    return th.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).mean()


def linearly_decaying_value(initial_value, decay_period, step, warmup_steps, final_value):
    """Returns the current value for a linearly decaying parameter.

    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.

    Args:
        decay_period: float, the period over which the value is decayed.
        step: int, the number of training steps completed so far.
        warmup_steps: int, the number of steps taken before the value is decayed.
        final value: float, the final value to which to decay the value parameter.

    Returns:
        A float, the current value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (initial_value - final_value) * steps_left / decay_period
    value = final_value + bonus
    value = np.clip(value, min(initial_value, final_value), max(initial_value, final_value))
    return value


def unique_tol(a: List[np.ndarray], tol=1e-4) -> List[np.ndarray]:
    """Returns unique elements of a list of np.arrays, within a tolerance."""
    if len(a) == 0:
        return a
    delete = np.array([False] * len(a))
    a = np.array(a)
    for i in range(len(a)):
        if delete[i]:
            continue
        for j in range(i + 1, len(a)):
            if np.allclose(a[i], a[j], tol):
                delete[j] = True
    return list(a[~delete])


def extrema_weights(dim: int) -> List[np.ndarray]:
    """Generate weight vectors in the extrema of the weight simplex. That is, one element is 1 and the rest are 0.

    Args:
        dim: size of the weight vector
    """
    return list(np.eye(dim, dtype=np.float32))


@lru_cache
def equally_spaced_weights(dim: int, n: int, seed: int = 42) -> List[np.ndarray]:
    """Generate weight vectors that are equally spaced in the weight simplex.

    It uses the Riesz s-Energy method from pymoo: https://pymoo.org/misc/reference_directions.html

    Args:
        dim: size of the weight vector
        n: number of weight vectors to generate
        seed: random seed
    """
    return list(get_reference_directions("energy", dim, n, seed=seed))


def random_weights(
    dim: int, n: int = 1, dist: str = "dirichlet", seed: Optional[int] = None, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Generate random normalized weight vectors from a Gaussian or Dirichlet distribution alpha=1.

    Args:
        dim: size of the weight vector
        n : number of weight vectors to generate
        dist: distribution to use, either 'gaussian' or 'dirichlet'. Default is 'dirichlet' as it is equivalent to sampling uniformly from the weight simplex.
        seed: random seed
        rng: random number generator
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if dist == "gaussian":
        w = rng.standard_normal((n, dim))
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1, keepdims=True)
    elif dist == "dirichlet":
        w = rng.dirichlet(np.ones(dim), n)
    else:
        raise ValueError(f"Unknown distribution {dist}")

    if n == 1:
        return w[0]
    return w


def log_episode_info(
    info: dict,
    scalarization,
    weights: Optional[np.ndarray],
    global_timestep: int,
    id: Optional[int] = None,
    writer: Optional[SummaryWriter] = None,
    verbose: bool = True,
):
    """Logs information of the last episode from the info dict (automatically filled by the RecordStatisticsWrapper).

    Args:
        info: info dictionary containing the episode statistics
        scalarization: scalarization function
        weights: weights to be used in the scalarization
        global_timestep: global timestep
        id: agent's id
        verbose: whether to print the episode info
    """
    episode_ts = info["l"]
    episode_time = info["t"]
    episode_return = info["r"]
    disc_episode_return = info["dr"]
    if weights is None:
        scal_return = scalarization(episode_return)
        disc_scal_return = scalarization(disc_episode_return)
    else:
        scal_return = scalarization(episode_return, weights)
        disc_scal_return = scalarization(disc_episode_return, weights)

    if verbose:
        print("Episode infos:")
        print(f"Steps: {episode_ts}, Time: {episode_time}")
        print(f"Total Reward: {episode_return}, Discounted: {disc_episode_return}")
        print(f"Scalarized Reward: {scal_return}, Discounted: {disc_scal_return}")

    if writer is not None:
        if id is not None:
            idstr = "_" + str(id)
        else:
            idstr = ""
        writer.add_scalar(f"charts{idstr}/timesteps_per_episode", episode_ts, global_timestep)
        writer.add_scalar(f"charts{idstr}/episode_time", episode_time, global_timestep)
        writer.add_scalar(f"metrics{idstr}/scalarized_episode_return", scal_return, global_timestep)
        writer.add_scalar(
            f"metrics{idstr}/discounted_scalarized_episode_return",
            disc_scal_return,
            global_timestep,
        )

        for i in range(episode_return.shape[0]):
            writer.add_scalar(
                f"metrics{idstr}/episode_return_obj_{i}",
                episode_return[i],
                global_timestep,
            )
            writer.add_scalar(
                f"metrics{idstr}/disc_episode_return_obj_{i}",
                disc_episode_return[i],
                global_timestep,
            )


def log_all_multi_policy_metrics(nl,
    current_front: List[np.ndarray],
    hv_ref_point: np.ndarray,
    reward_dim: int,
    global_step: int,
    n_sample_weights: int = 50,
    ref_front: Optional[List[np.ndarray]] = None,
):
    """Logs all metrics for multi-policy training.

    Logged metrics:
    - hypervolume
    - sparsity
    - expected utility metric (EUM)
    If a reference front is provided, also logs:
    - Inverted generational distance (IGD)
    - Maximum utility loss (MUL)

    Args:
        current_front (List) : current Pareto front approximation, computed in an evaluation step
        hv_ref_point: reference point for hypervolume computation
        reward_dim: number of objectives
        global_step: global step for logging
        n_sample_weights: number of weights to sample for EUM and MUL computation
        ref_front: reference front, if known
    """
    hv = hypervolume(hv_ref_point, current_front)
    sp = sparsity(current_front)
    eum = expected_utility(current_front, weights_set=equally_spaced_weights(reward_dim, n_sample_weights))

    nl.log_metric(metric_value=hv, metric_name='hypervolume',step=global_step , mode='metrics')
    nl.log_metric(metric_value=sp, metric_name="sparsity",step=global_step,mode='metrics')
    nl.log_metric(metric_value=eum, metric_name='enum',step=global_step, mode='metrics')

    # If PF is known, log the additional metrics
    if ref_front is not None:
        generational_distance = igd(known_front=ref_front, current_estimate=current_front)
        nl.log_metric(metric_value=generational_distance, metric_name='igd', step=global_step,mode='metrics')
        mul = maximum_utility_loss(
            front=current_front,
            reference_set=ref_front,
            weights_set=get_reference_directions("energy", reward_dim, n_sample_weights).astype(np.float32),
        )
        nl.log_metric(metric_value=mul, metric_name='maximum_utility_loss',step=global_step, mode='metrics')


def make_gif(env, agent, weight: np.ndarray, fullpath: str, fps: int = 50, length: int = 300):
    """Render an episode and save it as a gif."""
    assert "rgb_array" in env.metadata["render_modes"], "Environment does not have rgb_array rendering."

    frames = []
    state, info = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated) and len(frames) < length:
        frame = env.render()
        frames.append(frame)
        action = agent.eval(state, weight)
        state, reward, terminated, truncated, info = env.step(action)
    env.close()

    from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_gif(fullpath + ".gif", fps=fps)
    print("Saved gif at: " + fullpath + ".gif")


def seed_everything(seed: int):
    """Set random seeds for reproducibility.

    This function should be called only once per python process, preferably at the beginning of the main script.
    It has global effects on the random state of the python process, so it should be used with care.

    Args:
        seed: random seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = True


def load_yml(path_to_file):
    """

    Parameters
    ----------
    path_to_file

    Returns
    -------

    """
    with open(path_to_file) as f:
        file = yaml.safe_load(f)
    return file


def save_to_yml(data, name, save_path=''):
    """

    Parameters
    ----------
    data
    save_path
    """
    data_path = os.path.join(save_path, name)
    with open(data_path, 'w') as f:
        yaml.safe_dump(data, f)
    return data_path





def set_init(moduls):
    """

    Parameters
    ----------
    moduls
    """
    for modul in moduls:
        if modul is None:
            continue
        for layer in modul:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0., std=0.1)
                nn.init.constant_(layer.bias, 0.)


def init_layers(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)



def tensor_to_numpy(tensor):
    """

    Parameters
    ----------
    tensor

    Returns
    -------

    """
    if 'cuda' in tensor.device.type or 'mps' in tensor.device.type:
        tensor = tensor.cpu()
    return np.array(tensor)

def linear(r, pref, dim=-1):
    """

    Parameters
    ----------
    r
    pref
    r_
    dim

    Returns
    -------

    """
    return (r * pref).sum(dim=dim)


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


def pareto_filter(y):
    """

    Parameters
    ----------
    y

    Returns
    -------

    """
    is_pareto = [~((i < y).all(axis=1).any()) and ~((i == y).any(axis=1) & (i < y).any(axis=1)).any() for i in y]
    return np.unique(y[is_pareto], axis=0), np.arange(len(y))[is_pareto]

def tensor(arr, dtype=TORCH_PRECISION, device=None):
    """

    Parameters
    ----------
    arr
    dtype
    device

    Returns
    -------

    """
    if isinstance(arr, torch.Tensor):
        return arr
    if device is None:
        if torch.cuda.is_available() and USE_CUDA:
            return torch.tensor(arr, dtype=dtype).cuda()
        else:
            return torch.tensor(arr, dtype=dtype)
    else:
        return torch.tensor(arr, device=device, dtype=dtype)


def calc_hypervolume(y, utopia, antiutopia, n_samples=10000, rnd=True):
    """

    Parameters
    ----------
    y
    utopia
    antiutopia
    n_samples
    rnd

    Returns
    -------

    """
    if rnd:
        dist = uniform.Uniform(tensor(antiutopia), tensor(utopia))
        p = dist.sample([n_samples])
    else:
        p = tensor(multilinspace(utopia, antiutopia, n_samples))
    p_expand = p.unsqueeze(2).permute(2, 1, 0)
    y_expand = tensor(y.astype(float)).unsqueeze(2)
    return (p_expand < y_expand).all(dim=1).any(dim=0).sum() / float(n_samples)


def multilinspace(a, b, n):
    '''
    Linspace for multi-dimensional intervals.
    The input must be a list or an array.
    '''
    dim = len(a)
    if dim == 1:
        return np.linspace(a, b, n)

    n = int(np.floor(n ** (1. / dim)))
    tmp = []
    for i in range(dim):
        tmp.append(np.linspace(a[i], b[i], n))
    x = np.meshgrid(*tmp)
    y = np.zeros((n ** dim, dim))
    for i in range(dim):
        y[:, i] = x[i].flatten()
    return y


def calc_opt_reward(prefs, front, u_func=None):
    """

    Parameters
    ----------
    prefs
    front

    Returns
    -------

    """
    if u_func is None:
        u_func = lambda x, y: (x * y).sum(axis=1)
    prefs = np.float32(prefs)
    w_front = np.zeros(prefs.shape)
    for n, w in enumerate(prefs):
        id = u_func(front, w).argmax()
        w_front[n] = front[id]
    return w_front

