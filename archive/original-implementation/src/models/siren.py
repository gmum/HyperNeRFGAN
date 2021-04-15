"""
Copypasted from https://github.com/dalmia/siren
"""

from typing import List
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from .init import siren_uniform_


class SIREN(nn.Module):
    def __init__(self, config: Config):
        """
        SIREN model from the paper [Implicit Neural Representations with
        Periodic Activation Functions](https://arxiv.org/abs/2006.09661).

        :param layers: list of number of neurons in each hidden layer
        :type layers: List[int]
        :param in_features: number of input features
        :type in_features: int
        :param out_features: number of final output features
        :type out_features: int
        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        :param w0_initial: `w0` of first layer. defaults to 30 (as used in the paper)
        :type w0_initial: float, optional
        :param bias: whether to use bias or not. defaults to
            True
        :type bias: bool, optional
        :param initializer: specifies which initializer to use. defaults to 'siren'
        :type initializer: str, optional
        :param c: value used to compute the bound in the siren intializer.
            defaults to 6
        :type c: float, optional

        # References:
            -   [Implicit Neural Representations with Periodic Activation
                 Functions](https://arxiv.org/abs/2006.09661)
        """
        super(SIREN, self).__init__()

        layers = [config.layers_size] * config.num_layers

        self._check_params(layers)
        self.layers = [
            nn.Linear(config.in_features, layers[0], bias=config.has_bias),
            nn.BatchNorm1d(layers[0]) if config.has_bn else nn.Identity(),
            Sine(w0=config.w0_initial)]

        for index in range(len(layers) - 1):
            self.layers.extend([
                nn.Linear(layers[index], layers[index + 1], bias=config.has_bias),
                nn.BatchNorm1d(layers[index + 1]) if config.has_bn else nn.Identity(),
                Sine(w0=config.w0)
            ])

        self.layers.append(nn.Linear(layers[-1], config.out_features, bias=config.has_bias))
        self.model = nn.Sequential(*self.layers)

        if config.initializer is not None and config.initializer == 'siren':
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    siren_uniform_(m.weight, mode='fan_in', c=config.c)

    @staticmethod
    def _check_params(layers):
        assert isinstance(layers, list), 'layers should be a list of ints'
        assert len(layers) >= 1, 'layers should not be empty'

    def forward(self, x):
        return self.model(x)


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine activation function with w0 scaling support.
        Example:
            >>> w = torch.tensor([3.14, 1.57])
            >>> Sine(w0=1)(w)
            torch.Tensor([0, 1])
        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return torch.sin(self.w0 * x)

    @staticmethod
    def _check_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError('input to forward() must be torch.Tensor')
