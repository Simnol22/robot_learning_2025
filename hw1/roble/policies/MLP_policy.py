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
            self._mean_net = ptu.build_mlp(input_size=self._ob_dim,
                                      output_size=self._ac_dim,
                                      params=self._network)
            self._mean_net.to(ptu.device)

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
        obs_tensor = torch.FloatTensor(obs).to(ptu.device)
        with torch.no_grad():
            action_distribution = self.forward(obs_tensor)
        action = action_distribution.mean
        return action.cpu().numpy()
        

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
        return action_distribution

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
        pred_action_distribution = self.forward(obs_tensor)
        
        # Use mean for now
        loss = self._loss(pred_action_distribution.rsample(), action_tensor)
        
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
        # TODO: Create the full input to the IDM model (hint: it's not the same as the actor as it takes both obs and next_obs)
        obs_tensor = torch.FloatTensor(observations).to(ptu.device)
        next_obs = torch.FloatTensor(next_observations).to(ptu.device)
        action_tensor = torch.FloatTensor(actions).to(ptu.device)
        # TODO: Transform the numpy arrays to torch tensors (for obs, next_obs and actions)
        
        # TODO: Create the full input to the IDM model (hint: it's not the same as the actor as it takes both obs and next_obs)
        full_obs = torch.cat([obs_tensor, next_obs], dim=1)
        # TODO: Get the predicted actions from the IDM model (hint: you need to call the forward function of the IDM model)
        pred_action_distribution = self.forward(full_obs)
        # TODO: Compute the loss using the MLP_policy loss function
        loss = self._loss(pred_action_distribution.mean, action_tensor)
        loss.backward()
        # TODO: Update the IDM model.
        self._optimizer.step()
        return {
            'Training Loss IDM': ptu.to_numpy(loss),
        }