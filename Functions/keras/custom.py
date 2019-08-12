# -*- coding: utf-8 -*-
"""MilaX and MishX Activation Functions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine.base_layer import Layer
from keras import backend as K
from keras import initializers

class MilaX(Layer):
    '''
    MilaX Activation Function.
    .. math::
        MilaX(x) = x * tanh(ln(1 + e^{\\beta + x})) = x * tanh(softplus(\\beta + x)
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Parameter:
        - beta: scale to control the concavity of the global minima of the function. (Trainable Parameter)
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = MilaX()(X_input)
    '''

    def __init__(self, **kwargs):
        super(MilaX, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        # Create a trainable weight for beta parameter. Beta by default is initialized with -0.25.
        self.beta = self.add_weight(name='beta',
                                     initializer=initializers.Constant(value=-0.25),
                                     trainable=True,
                                     shape = (1,))
        super(MilaX, self).build(input_shape)

    def call(self, inputs):
        return inputs*K.tanh(K.softplus(inputs + self.beta))

    def get_config(self):
        base_config = super(MilaX, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class MishX(Layer):
    '''
    MishX activation function.
    .. math::
        \\MishX(x) = x * tanh(ln((1 + e^{x})^{\\beta}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Parameter:
        - beta: A trainable parameter
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = MishX()(X_input)
    '''
    
    def __init__(self, **kwargs):
        super(MishX, self).__init__(**kwargs)
        self.supports_masking = True
    
    def build(self, input_shape):
        # Create a trainable weight for beta parameter. Beta by default is initialized with 1.5.
        self.beta = self.add_weight(name='beta',
                                     initializer=initializers.Constant(value=1.5),
                                     trainable=True,
                                     shape = (1,))
        super(MishX, self).build(input_shape)
    
    def call(self, inputs):
        return inputs*K.tanh(K.log(K.pow((1+K.exp(inputs)),self.beta)))
        
    def get_config(self):
        base_config = super(MishX, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape
