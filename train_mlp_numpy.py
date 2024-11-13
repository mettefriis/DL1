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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    max_prediction_pr_datapoint=predictions.argmax(axis=1) #finding the highest softmax/logit
    correct_predictions=(max_prediction_pr_datapoint == targets) #list of correct predictions in the whole batch
    accuracy=correct_predictions.mean() #Mean of correct predictions

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    correct_counter=0
    samples_counter=0

    for inputs, targets in data_loader:
        predictions=model.forward(inputs) #before forward pass on batch
        batch_accuracy=accuracy(predictions, targets)

        n_correct=batch_accuracy * len(targets) #batch accurayc is a fraction. We want the raw number of correct predictions
        correct_counter+=n_correct
        samples_counter+=len(targets)

    avg_accuracy=correct_counter/samples_counter

    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    n_inputs = 32 * 32 * 3  # CIFAR-10 input size (32x32 image with 3 color channels)
    n_classes = 10 

    model = MLP(n_inputs, hidden_dims, n_classes) ###
    loss_module = CrossEntropyModule()
    # TODO: Training loop including validation
    epoch_counter=[]
    losses=[]
    val_accuracies = [] #list
    best_val_accuracy=0.0 #placeholder
    best_model = None #placeholder
    for epoch in range(epochs): #number of epochs given
        epoch_counter.append(epoch)
        for inputs, targets in cifar10_loader['train']:
              inputs = inputs.reshape(inputs.shape[0], -1)
              predictions=model.forward(inputs)
              loss =loss_module.forward(predictions, targets) #
              losses.append(loss) #add loss to list losses
              dout =loss_module.backward(predictions, targets) #calculate gradient
              model.backward(dout) #start the backward pass
        
        for layer in model.layers: #perform graident descent:
            if hasattr(layer, 'params'):
                for param in layer.params:
                    layer.params[param]-= lr*layer.grads[param]
                    
        val_accuracy=evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_accuracy)
        
        #update placeholders
        if val_accuracy > best_val_accuracy:
            best_val_accuracy=val_accuracy
            best_model=deepcopy(model)

          
    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])

    # TODO: Add any information you might want to save for plotting
    logging_dict = {
        'best_val_accuracy': best_val_accuracy,
        'val_accuracies': val_accuracies,
        'test_accuracies': test_accuracy,
        'losses':losses,
        'epoch':epoch_counter

    }
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here

    def plot_results(list_train, list_valid, name):
      fig = plt.figure(figsize=(8, 6), dpi=70)
      plt.xlabel('Number of Epochs', fontsize=12)
      plt.ylabel(name, fontsize=12)
      plt.plot(list_train, label="Training Set")
      plt.plot(list_valid, label="Validation Set")
      plt.legend()
      plt.show()

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    plot_results(logging_dict['test_accuracies'], logging_dict['val_accuracies'], 'Accuracy')






    
   



