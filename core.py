from typing import List, Optional, Tuple

import numpy as np
import scipy.signal
from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length: int, shape: Optional[Tuple[int, ...] | int]=None) -> Tuple[int, ...]:
    """
    Combine a length value with an optional shape to create a tuple.
    
    Args:
        length: An integer representing the first dimension of the output shape.
        shape: Optional shape specification. Can be an integer (scalar) or tuple of integers.
               If None, returns a 1D shape tuple.
    
    Returns:
        tuple: A tuple representing the combined shape. If shape is None, returns (length,).
               If shape is a scalar, returns (length, shape).
               If shape is a tuple, returns (length, *shape).
    
    Examples:
        >>> combined_shape(5)
        (5,)
        >>> combined_shape(5, 3)
        (5, 3)
        >>> combined_shape(5, (3, 4))
        (5, 3, 4)
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)  # type: ignore


def mlp(sizes: List[int], 
        activation=nn.Tanh, 
        output_activation=nn.Identity) -> nn.Sequential:
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module: nn.Module) -> int:
    """
    Count the total number of parameters in a PyTorch module.

    For each layer, we compute the product of the dimensions of its parameters.
    Then we sum the number of parameters across all layers to get the total count.
    """
    return sum([np.prod(p.shape) for p in module.parameters()]) # type: ignore


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


## Actor module is a MLP that outputs action distributions given observations.


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.

        # pi(.|obs): given a state, return the action distribution.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            # log π(a|s).
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        # π_θ(a | s) = N(μ_θ(s), σ)
        super().__init__()
        # -0.5 is just an initialization choice. 
        # It means that the initial standard deviation of the Gaussian distribution is exp(-0.5) ≈ 0.6065, 
        # which is a reasonable starting point for exploration.
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        # exp(log_std) is the standard deviation of the Gaussian distribution.
        # std is state-independent.
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # the output of mu_net serves as the mean of the Gaussian distribution. The standard deviation is a separate parameter (log_std).
        # mean is state-dependent.
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # log π(a|s) = -½·((a - μ)/σ)² - log σ - ½·log(2π)
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


## Critic module is a simple MLP that takes observations as input and 
# outputs values of state (V(s_t)).
# Note that it estimates the expected total discounted return from s_t onwards:
# V(s_t) = E[ R_t + γ R_{t+1} + γ^2 R_{t+2} + ... | s_t ]
# s_t --> MLP --> V(s_t)

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        """
        Perform a single step of the actor-critic model.

        Takes an observation and returns an action, value estimate, and log probability
        of the action under the current policy.

        Args:
            obs: Observation from the environment. Can be a tensor or array-like input
                 to the policy and value networks.

        Returns:
            tuple: A tuple containing:
                - a (numpy.ndarray): Action sampled from the policy distribution.
                - v (numpy.ndarray): Value estimate of the observation.
                - logp_a (numpy.ndarray): Log probability of the sampled action under
                                          the current policy distribution.

        Note:
            This method operates with torch.no_grad() to prevent gradient computation,
            making it suitable for inference/rollout collection.
        """
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        """
        Execute a single action based on the given observation.

        Args:
            obs: The observation input used to determine the action.

        Returns:
            The action to be taken based on the observation.
        """
        return self.step(obs)[0]