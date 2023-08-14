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
#from pymoo.util.ref_dirs import get_reference_directions

TORCH_PRECISION = torch.float32
TORCH_PRECISION_HIGH = torch.float64
USE_CUDA = True




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

