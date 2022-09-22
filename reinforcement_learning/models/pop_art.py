import torch
from torch import nn
from .model_utils import init
from torch.nn import functional as F

class PopArt(nn.Module):
    """
    Module Containing Pop-Art for reward normalization
    (Preserving Output Precision Adaptively Rescaling Targets)
    see: https://arxiv.org/pdf/1602.07714.pdf

    """
    def __init__(self, input_shape:int, output_shape: int, beta: float, eps=1e-5):
        super(PopArt, self).__init__()
        self.square_linear = nn.Linear(input_shape, output_shape)

        torch.nn.init.eye_(self.square_linear.weight)
        torch.nn.init.zeros_(self.square_linear.bias)

        
        # mean and std from data
        self.sigma = 1
        self.mu = 0

        self.eps = eps
        self.beta = beta


    def forward(self, x):
        """
        Normalize x 

        Args:
            x ([torch.tensor]):

        Returns:
            [torch.tensor]: Wx + b
        """
        return self.square_linear(x)

    def update(self, y):
        """
        Updated parameters:
        W and b before gradient descent step
        mu and sigma to new values according to a expontential moving average of targets

        Args:
            y ([torch.tensor]): targets
        """
        new_mu = (1 - self.beta) * self.mu + self.beta * y.mean()
        new_sigma = ((1 - self.beta) * self.sigma ** 2 + self.beta * (y ** 2).mean()) ** 0.5
        new_prec = 1.0 / (new_sigma + self.eps)
        with torch.no_grad():
            self.square_linear.weight.copy_(new_prec * self.sigma * self.square_linear.weight)
            self.square_linear.bias.copy_(new_prec * (self.sigma * self.square_linear.bias + self.mu - new_mu))
            self.mu = new_mu
            self.sigma = new_sigma

    def output(self ,x):
        """
        Return unormalized input by scaling and un-scaling
        x = sigma(Wx+b) + mu

        Args:
            x ([torch.tensor]): input

        Returns:
            [torch.tensor]: unormalized input
        """
        x = self.square_linear(x)
        return self.sigma * x + self.mu

    def normalize(self, y):
        """
        Return standard normalization
        """
        return (y - self.mu) / (self.sigma + self.eps)