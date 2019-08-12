'''
Script implements MishX activation:

.. math::
    MishX(x) = x * tanh(ln((1 + e^{x})^{\\beta}))
'''
 # import standard libraries
import numpy as np

# import torch
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class MishX(nn.Module):
    '''
    Implementation of MishX activation:

        .. math::

        Mish(x) = x * tanh(ln((1 + e^{x})^{\\beta}))

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
            beta is initialized with 1.5 value by default
        '''
        super(MishX,self).__init__()
        self.in_features = in_features

        # initialize alpha
        if beta == None:
            self.beta = Parameter(torch.tensor(1.5)) # create a tensor out of beta
        else:
            self.beta = Parameter(torch.tensor(alpha)) # create a tensor out of beta

        self.alpha.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        '''
        Forward pass of the function
        '''
        return x * torch.tanh(torch.log(torch.pow((1+torch.exp(x)),self.beta)))