import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.env_params = kwargs

        self._learn_policy_std = self.env_params['learn_policy_std']

        if self._discrete:
            self._logits_na = ptu.build_mlp(input_size=self._ob_dim,
                                           output_size=self._ac_dim,
                                           params=self._network)
            self._logits_na.to(ptu.device)
            self._mean_net = None
            self._logstd = None
            self._optimizer = optim.Adam(self._logits_na.parameters(),
                                        self._learning_rate)
        else:
            self._logits_na = None
            self._mean_net = ptu.build_cnn(input_size=64*64,
                                        output_size=self._ac_dim,
                                        params=self._network)

            self._mean_net.to(ptu.device)
            self._cnn_net = ptu.build_cnn(input_size=64*64,
                                        output_size=self._ac_dim,
                                        params=self._network)

            if self._deterministic:
                self._optimizer = optim.Adam(
                    itertools.chain(self._mean_net.parameters()),
                    self._learning_rate
                )
            else:
                self._std = nn.Parameter(
                    torch.ones(self._ac_dim, dtype=torch.float32, device=ptu.device) * 0.1
                )
                self._std.to(ptu.device)
                if self._learn_policy_std:
                    self._optimizer = optim.Adam(
                        itertools.chain([self._std], self._mean_net.parameters()),
                        self._learning_rate
                    )
                else:
                    self._optimizer = optim.Adam(
                        itertools.chain(self._mean_net.parameters()),
                        self._learning_rate
                    )

        if self._nn_baseline:
            self._baseline = ptu.build_mlp(
                input_size=self._ob_dim,
                output_size=1,
                params=self._network
            )
            self._baseline.to(ptu.device)
            self._baseline_optimizer = optim.Adam(
                self._baseline.parameters(),
                self._critic_learning_rate,
            )
        else:
            self._baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self._deterministic:
            if len(obs.shape) > 1:
                observation = obs
            else:
                observation = obs[None, :]
            observation = ptu.from_numpy(observation.astype(np.float32))
            action = self.forward(observation)
            return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self._discrete:
            logits = self._logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            if self._deterministic:
                return self._mean_net(observation)
            else:
                action_distribution = distributions.Normal(self._mean_net(observation), self._std)
                action = action_distribution.rsample()
        return action

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loss = nn.MSELoss()

    def update(
        self, observations, actions,
        adv_n=None, acs_labels_na=None, qvals=None
        ):
        self._optimizer.zero_grad()

        obs_tensor = torch.FloatTensor(observations).to(ptu.device)
        action_tensor = torch.FloatTensor(actions).to(ptu.device)

        # Forward pass to get pred action distribution
        pred_action = self.forward(obs_tensor)
        
        loss = self._loss(pred_action, action_tensor)
        
        loss.backward()
        self._optimizer.step()
        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_idm(
        self, observations, actions, next_observations,
        adv_n=None, acs_labels_na=None, qvals=None
        ):
        self._optimizer.zero_grad()

        obs_tensor = torch.FloatTensor(observations).to(ptu.device)
        next_obs = torch.FloatTensor(next_observations).to(ptu.device)
        action_tensor = torch.FloatTensor(actions).to(ptu.device)

        full_obs = torch.cat([obs_tensor, next_obs], dim=1)

        pred_action = self.forward(full_obs)

        loss = self._loss(pred_action, action_tensor)

        loss.backward()
        self._optimizer.step()
        return {
            'Training Loss IDM': ptu.to_numpy(loss),
        }