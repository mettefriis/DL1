################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization.
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #1) Initialize weight parameters using Kaiming initialization 
        limit = np.sqrt(5 / in_features) if input_layer else np.sqrt(2 / in_features) #This is how it is usually done with kaiming initlaisation.
        self.params = {
            'weight': np.random.uniform(-limit, limit, (out_features, in_features)),
            'bias': np.zeros(out_features)  # 2) Initialize biases with zeros
        }

        self.grads = {
        'weight': np.zeros((out_features, in_features)),
        'bias': np.zeros(out_features)
        }

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.input=x #self points to x as input
        x = x.reshape(x.shape[0], -1)
        out=np.dot(x, self.params['weight'].T) + self.params['bias'] #dot product

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads['weight']=np.dot(dout.T, self.input) #grad of L wrt. W
        self.grads['bias']= dout.sum(axis=0) #grad of L wrt. b
        dx = np.dot(dout, self.params['weight']) #grad of L wrt. X

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #i
        #######################
        
        self.input=None #erase memory of input
        self.output=None #erase memory of output
        pass
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.input=x
        out=np.where(x>=0, x, self.alpha*np.exp(x)-1)

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        dx = np.copy(dout)
        dx[self.input >= 0] = dout[self.input >= 0]
        dx[self.input < 0] = dout[self.input < 0] * self.alpha * np.exp(self.input[self.input < 0])


        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.input=None
        self.output=None
        pass
    
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        #Find max for each row
        #Choose that, and choose dimension of x:

        max_x = x.max(1)
        max_x = np.reshape(max_x, (max_x.shape[0], 1))
        exp = np.exp(x - max_x)
        self.out = exp/np.sum(exp, axis = -1)[:,None]

        #######################
        # END OF YOUR CODE    #
        #######################

        return self.out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        #set y to the output "out" from before:
        y=self.out

        #calculate the gradient:
        dx = y * (dout - (dout * y)@ np.ones((y.shape[1],y.shape[1]))) ##!!

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.output=None

        pass
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #Convert target labels to one-hot:
        # Assuming y has shape (128,) with class labels like [0, 1, ..., 9]
        y_one_hot = np.zeros((y.size, x.shape[1]))  # Create one-hot encoded y
        y_one_hot[np.arange(y.size), y] = 1          # Populate one-hot matrix
        out = -np.trace(y_one_hot @ np.log(x).T) / x.shape[0]  # Cross-entropy calculation

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        y_one_hot = np.zeros((y.size, x.shape[1]))  # Create one-hot encoded y
        y_one_hot[np.arange(y.size), y] = 1          # Populate one-hot matrix
        dx=-(y_one_hot)/x.shape[0]/x ### !

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx