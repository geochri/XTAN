'''
Script implements MilaX activation:

.. math::
    MilaX(x) = x * tanh(softplus(\\beta + x)) = x * tanh(ln(1 + e^{\\beta + x}))
'''
 # import standard libraries
import numpy as np

# import torch
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class MilaX(nn.Module):
    '''
    Implementation of MilaX activation:

        .. math::

        MilaX(x) = x * tanh(softplus(\\beta + x)) = x * tanh(ln(1 + e^{\\beta + x}))

    with trainable parameter beta.

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Parameters:
        - beta - trainable parameter
    '''
    def __init__(self, in_features, beta = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - beta: learnable parameter
            beta is initialized with -0.25 value by default
        '''
        super(MilaX,self).__init__()
        self.in_features = in_features

        # initialize alpha
        if beta == None:
            self.beta = Parameter(torch.tensor(-0.25)) # create a tensor out of beta
        else:
            self.beta = Parameter(torch.tensor(alpha)) # create a tensor out of beta

        self.alpha.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        '''
        Forward pass of the function
        '''
        return x * torch.tanh(F.softplus(x + self.beta))
