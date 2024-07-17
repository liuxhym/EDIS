import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.network import MLP, ResNetPreActivation, ResNetPreActivationLayer
import time

class energy_model(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dims,
            activation,
            spectral_norm: bool,
            with_reward: bool = False,
            layer_type: str = "MLP",
            device: str = "cpu",
            timestep: bool = True
    ):
        super().__init__()

        self.device = torch.device(device)
        self.activation = activation
        self.with_reward = True

        def get_dense_net(*args):
            dense_net = nn.Linear(*args)
            if spectral_norm:
                dense_net = nn.utils.spectral_norm(dense_net)
            return dense_net

        if layer_type == "MLP":  
            self.mlp = MLP(2 * obs_dim + action_dim + with_reward + timestep, hidden_dims, 1, dropout_rate=None)
            self.project_energy = nn.Identity()

        elif layer_type == "ResNetPreActivation":
            mlp_module = [ get_dense_net( 2 * obs_dim + action_dim + with_reward + timestep, hidden_dims[0]) ]
            # if spectral_norm:
            #     self.project_resnet = nn.utils.spectral_norm(self.project_resnet)
            for l in range(len(hidden_dims)-1):
                mlp_module.append(ResNetPreActivationLayer(hidden_dims[l], hidden_dims[l+1]))
            # self.mlp = nn.ModuleList(mlp_module)
            self.mlp = nn.Sequential(*mlp_module)
            self.project_energy = get_dense_net(hidden_dims[-1], 1)

        self.to(self.device)

    def forward(self, transition):
        input = self.mlp(transition)
        output = self.project_energy(input)
        return output.squeeze(axis=-1)

