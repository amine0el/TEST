"""MORL algorithm base classes."""
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import gymnasium as gym
import numpy as np
import torch as th

from gymnasium import spaces
from mo_gymnasium.utils import MOSyncVectorEnv
from torch.utils.tensorboard import SummaryWriter
from neptune_logger import NeptuneLogger

from evaluation import (
    eval_mo_reward_conditioned,
    policy_evaluation_mo,
)


class MOPolicy(ABC):
    """An MORL policy.

    It has an underlying learning structure which can be:
    - used to get a greedy action via eval()
    - updated using some experiences via update()

    Note that the learning structure can embed multiple policies (for example using a Conditioned Network).
    In this case, eval() requires a weight vector as input.
    """

    def __init__(self, id: Optional[int] = None, device: Union[th.device, str] = "auto") -> None:
        """Initializes the policy.

        Args:
            id: The id of the policy
            device: The device to use for the tensors
        """
        self.id = id
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device == "auto" else device
        self.global_step = 0

    @abstractmethod
    def eval(self, obs: np.ndarray, w: Optional[np.ndarray]) -> Union[int, np.ndarray]:
        """Gives the best action for the given observation.

        Args:
            obs (np.array): Observation
            w (optional np.array): weight for scalarization

        Returns:
            np.array or int: Action
        """

    def __report(
        self,
        scalarized_return,
        scalarized_discounted_return,
        vec_return,
        discounted_vec_return,
        nl: NeptuneLogger,
    ):
        if self.id is None:
            idstr = ""
        else:
            idstr = f"_{self.id}"
        nl.log_metric(metric_value=scalarized_return,metric_name=f"eval{idstr}/scalarized_return",step=self.global_step, mode='metrics')
        nl.log_metric(metric_value=scalarized_discounted_return,metric_name=f"eval{idstr}/scalarized_discounted_return",step=self.global_step,mode='metrics')
        for i in range(vec_return.shape[0]):
            nl.log_metric(metric_value=vec_return[i],metric_name=f"eval{idstr}/vec_{i}",step=self.global_step,mode='metrics')
            nl.log_metric(metric_value=discounted_vec_return[i],metric_name=f"eval{idstr}/discounted_vec_{i}",step=self.global_step,mode='metrics')
          

    def policy_eval(
        self,
        eval_env,
        num_episodes: int = 5,
        scalarization=np.dot,
        weights: Optional[np.ndarray] = None,
        writer: Optional[SummaryWriter] = None,
    ):
        """Runs a policy evaluation (typically over a few episodes) on eval_env and logs some metrics using writer.

        Args:
            eval_env: evaluation environment
            num_episodes: number of episodes to evaluate
            scalarization: scalarization function
            weights: weights to use in the evaluation

        Returns:
             a tuple containing the average evaluations
        """
        (
            scalarized_return,
            scalarized_discounted_return,
            vec_return,
            discounted_vec_return,
        ) = policy_evaluation_mo(self, eval_env, w=weights, rep=num_episodes)

        if writer is not None:
            self.__report(
                scalarized_return,
                scalarized_discounted_return,
                vec_return,
                discounted_vec_return,
                writer,
            )

        return scalarized_return, scalarized_discounted_return, vec_return, discounted_vec_return

    def policy_eval_esr(
        self,
        eval_env,
        scalarization,
        weights: Optional[np.ndarray] = None,
        writer: Optional[SummaryWriter] = None,
    ):
        """Runs a policy evaluation (typically on one episode) on eval_env and logs some metrics using writer.

        Args:
            eval_env: evaluation environment
            scalarization: scalarization function
            weights: weights to use in the evaluation

        Returns:
             a tuple containing the evaluations
        """
        (
            scalarized_reward,
            scalarized_discounted_reward,
            vec_reward,
            discounted_vec_reward,
        ) = eval_mo_reward_conditioned(self, eval_env, scalarization, weights)

        if writer is not None:
            self.__report(
                scalarized_reward,
                scalarized_discounted_reward,
                vec_reward,
                discounted_vec_reward,
                writer,
            )

        return scalarized_reward, scalarized_discounted_reward, vec_reward, discounted_vec_reward

    @abstractmethod
    def update(self) -> None:
        """Update algorithm's parameters (e.g. using experiences from the buffer)."""


class MOAgent(ABC):
    """An MORL Agent, can contain one or multiple MOPolicies. Contains helpers to extract features from the environment, setup logging etc."""

    def __init__(self, env: Optional[gym.Env], device: Union[th.device, str] = "auto", seed: Optional[int] = None) -> None:
        """Initializes the agent.

        Args:
            env: (gym.Env): The environment
            device: (str): The device to use for training. Can be "auto", "cpu" or "cuda".
            seed: (int): The seed to use for the random number generator
        """
        self.extract_env_info(env)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device == "auto" else device

        self.global_step = 0
        self.num_episodes = 0
        self.seed = seed
        self.np_random = np.random.default_rng(self.seed)

    def extract_env_info(self, env: Optional[gym.Env]) -> None:
        """Extracts all the features of the environment: observation space, action space, ...

        Args:
            env (gym.Env): The environment
        """
        # Sometimes, the environment is not instantiated at the moment the MORL algorithms is being instantiated.
        # So env can be None. It is the responsibility of the implemented MORLAlgorithm to call this method in those cases
        if env is not None:
            self.env = env
            if isinstance(self.env.observation_space, spaces.Discrete):
                self.observation_shape = (1,)
                self.observation_dim = self.env.observation_space.n
            else:
                self.observation_shape = self.env.observation_space.shape
                self.observation_dim = self.env.observation_space.shape[0]

            self.action_space = env.action_space
            if isinstance(self.env.action_space, (spaces.Discrete, spaces.MultiBinary)):
                self.action_shape = (1,)
                self.action_dim = self.env.action_space.n
            else:
                self.action_shape = self.env.action_space.shape
                self.action_dim = self.env.action_space.shape[0]
            self.reward_dim = self.env.reward_space.shape[0]

    @abstractmethod
    def get_config(self) -> dict:
        """Generates dictionary of the algorithm parameters configuration.

        Returns:
            dict: Config
        """

