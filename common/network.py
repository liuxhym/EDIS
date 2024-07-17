import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + [256] * hidden_dims
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNetPreActivationLayer(nn.Module):
    def __init__(            
            self,
            input_output_dim,
            hidden_dim,
            activation=nn.ReLU,
            normalization=nn.LayerNorm,
            dropout=0.1,
            ):
        super().__init__()

        self.input_dim = self.output_dim = input_output_dim
        self.hidden_dim = hidden_dim

        self.norm_layer_1 = normalization(input_output_dim)
        self.activation_layer_1 = activation()
        self.dropout_1 = nn.Dropout(dropout) if dropout else None
        self.linear_layer_1 = nn.Linear(input_output_dim, hidden_dim)
        
        self.norm_layer_2 = normalization(hidden_dim)
        self.activation_layer_2 = activation()
        self.dropout_2 = nn.Dropout(dropout) if dropout else None
        self.linear_layer_2 = nn.Linear(hidden_dim, input_output_dim)

    def forward(self, x):
        y = self.norm_layer_1(x)
        y = self.activation_layer_1(y)
        if self.dropout_1 is not None:
            y = self.dropout_1(y) 
        y = self.linear_layer_1(y)

        y = self.norm_layer_2(y)
        y = self.activation_layer_2(y)
        if self.dropout_2 is not None:
            y = self.dropout_2(y)
        y = self.linear_layer_2(y)
        return x + y


class ResNetPreActivation(nn.Module):
    def __init__(
            self,
            input_dim,
            out_dim,
            res_dim,
            res_hidden_dim,
            n_res_layers,
            activation=nn.ReLU,
            normalization=nn.LayerNorm,
            dropout=0.1,
            device="cpu",
            ):
        super().__init__()
        self.device = torch.device(device)
        self.projection_layer = nn.Linear(input_dim, res_dim)
        self.projection_output = nn.Linear(res_dim, out_dim)

        module_list = [ self.projection_layer ]

        for l in range(n_res_layers):
            module_list.append(ResNetPreActivationLayer(res_dim, res_hidden_dim, activation, normalization, dropout))

        module_list.append(self.projection_output)
        self.backbones = nn.ModuleList(module_list)
        self.to(self.device)

    def forward(self, x):
        for layer in self.backbones:
            x = layer(x)
        return x