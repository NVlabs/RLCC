import torch
from torch import nn
import numpy as np
from typing import List, Tuple
from .model_utils import init
import math
from torch.nn import functional as F

ACTIVATION = dict(relu=nn.ReLU(), tanh=nn.Tanh())

from lut_activation import sig_table, tanh_table
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules import _utils

class MLPAMAX(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation_function: str, hidden_sizes: List[int], use_lstm: bool=False):
        super(MLPAMAX, self).__init__()
        self.activation_function = ACTIVATION[activation_function]
        self.hidden_sizes = hidden_sizes

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        layers = []
        previous_dim = input_size
        for dim in hidden_sizes:
            layers.append(init_(nn.Linear(previous_dim, dim, bias=False)))
            layers.append(self.activation_function)
            previous_dim = dim

        self.lstm = None
        if use_lstm:
            self.lstm = LSTMCell(previous_dim, previous_dim, bias=False)

        self.net = nn.Sequential(*layers)
        self.output_layer = init_(nn.Linear(previous_dim, output_size, bias=False))


    def forward(self, x: torch.tensor,  hc: Tuple[torch.tensor, torch.tensor] = None ) -> Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        x = self.net(x)
        if self.lstm is not None:
            hc = self.lstm(x, hc)
            h, c = hc
            x = h
        x = self.output_layer(x)
        return x, hc

class LSTMCell(nn.Module, _utils.QuantMixin):
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def __init__(self, input_size: int, hidden_size: int, bias: bool, **kwargs):
        super(LSTMCell, self).__init__()
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        self.init_quantizer(quant_desc_input, quant_desc_weight)

        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.fc=nn.Linear(input_size, hidden_size, bias=False)
        # self.fc_ih = nn.Linear(12,48, bias=False)
        # self.fc_hh = nn.Linear(12,48, bias=False)

        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.sig_table = sig_table()
        self.tanh_table = tanh_table()


    def forward(self, input, hx=None):
        #check inputs
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        hx, cx = hx
        if self._input_quantizer is not None:
            input, hx = self._input_quantizer(torch.cat([input, hx], 1)).split([input.size()[1], hx.size()[1]], 1)
        if self._weight_quantizer is not None:
            w_ih, w_hh = self._weight_quantizer(torch.cat([self.weight_ih, self.weight_hh], 1)).split([self.weight_ih.size()[1], self.weight_hh.size()[1]], 1)
        gates = F.linear(input, self.weight_ih) + F.linear(hx, self.weight_hh)
        # gates = self.fc_ih(input) + self.fc_hh(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = self.sig_table.get_value(ingate)
        forgetgate = self.sig_table.get_value(forgetgate)
        cellgate = self.tanh_table.get_value(cellgate)
        outgate = self.sig_table.get_value(outgate)
        # ingate = torch.sigmoid(ingate)
        # forgetgate = torch.sigmoid(forgetgate)
        # cellgate = torch.tanh(cellgate)
        # outgate = torch.sigmoid(outgate)
        cy = (forgetgate * cx) + (ingate * cellgate)
        # hy = outgate * torch.tanh(cy)
        hy = outgate * self.tanh_table.get_value(cy)
        return hy, cy


    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))


