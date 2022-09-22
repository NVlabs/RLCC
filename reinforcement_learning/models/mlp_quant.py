import torch
from torch import nn
import numpy as np
from typing import List, Tuple
from .model_utils import init
import math
from torch.nn import functional as F

ACTIVATION = dict(relu=nn.ReLU(), tanh=nn.Tanh())


class MLPQ(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation_function: str, hidden_sizes: List[int], use_lstm: bool=False):
        super(MLPQ, self).__init__()
        self.activation_function = ACTIVATION[activation_function]
        self.hidden_sizes = hidden_sizes

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        layers = []
        previous_dim = input_size
        h_dim =hidden_sizes[0]
        self.fc_1 = Linear(previous_dim, h_dim)
        self.relu = self.activation_function
        self.lstm = None
        if use_lstm :
            self.lstm = MLSTMCell(h_dim, h_dim)
        # self.net = nn.Sequential(*layers)
        self.output_layer = Linear(h_dim, output_size)
        self.param_dict = None

    def forward(self, x: torch.tensor,  hc: Tuple[torch.tensor, torch.tensor] = None ) -> Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        x = self.quant_tensor(x, self.param_dict['net.0._input_quantizer._amax'])
        # x = self.fc_1.forward(x, self.param_dict['net.0.weight'],  self.param_dict['net.0.bias']) #TODO fix no bias
        x = self.fc_1.forward(x, self.param_dict['net.0.weight'],  None) #TODO fix
        #TODO should I move it to 8?
        x = self.relu(x)
        x = self.quant_tensor(self.dequant_tensor(x, self.param_dict['net.0._input_quantizer._amax'] * self.param_dict['net.0._weight_quantizer._amax'],  double=True), self.param_dict['lstm._input_quantizer._amax'])
        if self.lstm is not None:
            #qunat hx
            #add if type not int 8 do the qunat.
            if hc[0].dtype != torch.int8:
                hc = torch.cat(hc)
                h, c = self.quant_tensor(hc, self.param_dict['lstm._input_quantizer._amax']).chunk(2)
                hc = [h, c]
            # hc = self.lstm(x, hc, self.param_dict['lstm._input_quantizer._amax']* self.param_dict['lstm._weight_quantizer._amax'], self.param_dict['lstm.weight_ih'], self.param_dict['lstm.weight_hh'], self.param_dict['lstm.bias_ih'], self.param_dict['lstm.bias_hh']) #TODO change
            hc = self.lstm(x, hc, self.param_dict['lstm._input_quantizer._amax']* self.param_dict['lstm._weight_quantizer._amax'], self.param_dict['lstm.weight_ih'], self.param_dict['lstm.weight_hh'], None, None) #TODO change
            h, c = hc
            x = h
        #quant x from int 32 to int 8
        x = self.quant_tensor(x, self.param_dict['output_layer._input_quantizer._amax'])
        # x = self.output_layer.forward(x, self.param_dict['output_layer.weight'], self.param_dict['output_layer.bias'])
        x = self.output_layer.forward(x, self.param_dict['output_layer.weight'], None)

        #output should be in int32
        #TODO what to do withoutput
        x = self.dequant_tensor(x, self.param_dict['output_layer._input_quantizer._amax'] * self.param_dict['output_layer._weight_quantizer._amax'], double=True)

        return x, hc

    def quant_tensor(self, input, amax):
        scale = 127/amax
        output = torch.clamp((input * scale).round_(), -127, 127)
        return output.type(torch.int8)

    def dequant_tensor(self, input, amax, double=False):
        max_range = 127 if not double else 127**2
        scale = max_range/amax
        if len(scale.shape)> 1:
            scale = scale.squeeze(1)
        return input/scale

class Linear():
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.in_features = in_features
        self.out_features = out_features


    def forward(self, input, weight, bias):
        input = input.type(torch.int32)
        weight = weight.type(torch.int32)
        # bias = bias.type(torch.int32)
        # return input.matmul(weight.t()) + bias
        return input.matmul(weight.t())


class MLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(MLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc=Linear(64,64)

    def forward(self, input, hx=None, amax=None, weight_ih=None, weight_hh=None, bias_ih=None, bias_hh=None):
        #check inputs
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        hx, cx = hx
        #RUN
        gates = self.fc.forward(input, weight_ih, bias_ih) + self.fc.forward(hx, weight_hh,  bias_hh)
        if amax is not None:
            gates = self.dequant_tensor(gates, amax, double=True)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        gates = torch.cat([ingate, forgetgate, cellgate, outgate], axis=1) #cat to a matrix
        gates = self.quant_tensor(gates, amax) #need to be int32 for output
        gates = gates.type(torch.int32)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        cy_1 = forgetgate * cx
        cy_2 = ingate * cellgate

        cy = self.dequant_tensor(cy_1, amax[16:32] * 3.4764, double=True) + self.dequant_tensor(cy_2, amax[:16] * amax[32:48], double=True)
        tanh_cy = torch.tanh(cy)
        outgate = self.dequant_tensor(outgate, amax[48:])
        hy = outgate * torch.tanh(cy)
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


    def dequant_tensor(self, input, amax, double=False):
        max_range = 127 if not double else 127**2
        scale = max_range/amax
        if len(scale.shape)> 1:
            scale = scale.squeeze(1)
        return input/scale

    def quant_tensor(self, input, amax, double=False):
        max_range = 127 if not double else 127**2
        scale = (max_range/amax).squeeze(1)
        output = torch.clamp((input * scale).round_(), -127, 127)
        return output.type(torch.int8)