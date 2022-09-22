from numpy.core.numeric import ones
import torch
from torch import nn
import numpy as np
from typing import List, Tuple
from .model_utils import init
import math
from torch.nn import functional as F

RNN_TO_MODEL = {'LSTM': nn.LSTMCell, 'GRU': nn.GRUCell, 'RNN': nn.RNNCell}
class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int,  lrelu_coeff: float, hidden_sizes: List[int], use_rnn: str=None, bias=False,
     device: torch.device=torch.device('cpu')):
        super(MLP, self).__init__()
        self.hidden_sizes = hidden_sizes

        self.bias = bias
        self.device = device

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        layers = []
        previous_dim = input_size
        for dim in hidden_sizes:
            layers.append(init_(nn.Linear(previous_dim, dim, bias=bias)))
            activation = nn.LeakyReLU(lrelu_coeff) if lrelu_coeff > 0 else nn.ReLU()
            layers.append(activation)
            previous_dim = dim

        self.rnn = None
        if use_rnn:
            # we want the bias to be learned only for the input
            self.rnn = RNN_TO_MODEL[use_rnn](previous_dim, previous_dim, bias=bias)
        self.rnn_type = use_rnn

        self.net = nn.ModuleList(layers)

        # self.output_layer = init_(nn.Linear(previous_dim, output_size))
        self.output_layer = init_(nn.Linear(previous_dim, output_size, bias=bias))


    def forward(self, x: torch.tensor,  hc: Tuple[torch.tensor, torch.tensor] = None ) -> Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        # print(f'input: {x}')
        for layer in self.net:
            x = layer(x)
            # print(f'input_layer: {x}')
        if self.rnn is not None:
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            if self.rnn_type == 'LSTM':
                hc = self.rnn(x, hc)
                # print(f'lstm output: {x}')
            else:
                if hc is None:
                    hc = self.rnn(x, hc)
                    # print(f'rnn output: {x}')
                else:
                    hc = self.rnn(x, hc[0])
                    # print(f'rnn output from previous hc: {x}')
                hc = (hc, hc)
            x = hc[0]
        x = self.output_layer(x)
        # print(f'final output: {x}')
        return x, hc

