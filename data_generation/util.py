"""
This file contains the following important contents:

    AcasNetworkParser: A parser for the ACAS XU networks in nnet format.
                       The code was taken from Rudy Bunel's github
                       https://github.com/oval-group/PLNN-verification/plnn/model.py
                       adjusted by Dominik Winterer and Ying Wang to our needs


    FeedForwardNeuralNet: A simple fully connected feedforward DNN
                          able to read in the parameters of an ACAS XU net.

"""
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from numpy import genfromtxt

import pandas as pd
import numpy as np
import math

from collections import Counter

import logging
import sys
FORMAT = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)


class AcasNetworkParser:
    def __init__(self, infile):
        self.filename = infile.name

        def readline(): return infile.readline().strip()
        line = readline()

        # Ignore the comments
        while line.startswith('//'):
            line = readline()

        # Parse the dimensions
        all_dims = [int(dim) for dim in line.split(',')
                    if dim != '']
        self.num_layers, self.input_size, \
        self.output_size, self.max_lay_size = all_dims

        # Get the layers size
        line = readline()
        self.nodes_in_layer = [int(l_size_str) for l_size_str
                               in line.split(',') if l_size_str != '']

        assert self.input_size == self.nodes_in_layer[0]
        assert self.output_size == self.nodes_in_layer[-1]

        # Load the symmetric parameter
        line = readline()
        # is_symmetric = int(line.split(',')[0]) != 0
        # if symmetric == 1, enforce that psi (input[2]) is positive
        # if to do so, it needs to be flipped, input[1] is also adjusted
        # In practice, all the networks released with Reluxplex 1.0 have
        # it as 0 so we will just ignore it.

        # Load Min/Max/Mean/Range values of inputs
        line = readline()
        self.inp_mins = [float(min_str) for min_str in line.split(',')
                         if min_str != '']

        line = readline()
        self.inp_maxs = [float(max_str) for max_str in line.split(',')
                         if max_str != '']
        line = readline()
        self.inpout_means = [float(mean_str) for mean_str in line.split(',')
                             if mean_str != '']

        line = readline()
        self.inpout_ranges = [float(range_str) for range_str in line.split(',')
                              if range_str != '']

        assert len(self.inp_mins) == len(self.inp_maxs)
        assert len(self.inpout_means) == len(self.inpout_ranges)
        assert len(self.inpout_means) == (len(self.inp_mins) + 1)

        # Load the weights
        self.parameters = []
        for layer_idx in range(self.num_layers):
            # Gather weight matrix
            weights = []
            biases = []
            for neuron in range(self.nodes_in_layer[layer_idx + 1]):
                line = readline()
                to_neuron_weights = [float(wgt_str) for wgt_str
                                     in line.split(',') if wgt_str != '']
                assert len(to_neuron_weights) == self.nodes_in_layer[layer_idx]
                weights.append(to_neuron_weights)
            for neuron in range(self.nodes_in_layer[layer_idx + 1]):
                line = readline()
                neuron_biases = [float(bias_str) for bias_str
                                 in line.split(',') if bias_str != '']
                assert len(neuron_biases) == 1
                biases.append(neuron_biases[0])
            assert len(weights) == len(biases)
            self.parameters.append((weights, biases))

    def to_torch(self):
        # Return a fully connected neural network with the architecture and
        # parameters as specified
        neural_net = FeedForwardNeuralNet(self.nodes_in_layer)
        neural_net.set_parameters(self.parameters)
        return neural_net

    def dump(self):
        print("")
        print("Filename: " + str(self.filename))
        print("Number of layers: " + str(self.num_layers))
        print("Input size: " + str(self.input_size))
        print("Ouput size: " + str(self.output_size))
        print("Max layer size: " + str(self.max_lay_size))
        print("Num neurons per layer: " + str(self.nodes_in_layer))
        print("Minimum values of inputs: ", end="")
        print(self.inp_mins)
        print("Maximum values of inputs: ", end="")
        print(self.inp_maxs)
        # Understand what those two parameters actually mean
        print("Inpout means: ", end="")
        print(self.inpout_means)
        print("Inpout ranges: ", end="")
        print(self.inpout_ranges)


class FeedForwardNeuralNet(nn.Module):
    def __init__(self, layer_sizes):
        super(FeedForwardNeuralNet, self).__init__()
        self.hidden = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.hidden.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return x

    def set_parameters(self, parameters):
        i = 0
        for name, param in self.named_parameters():
            if "weight" in name:
                param.data = torch.tensor(parameters[i][0])
            if "bias" in name:
                param.data = torch.tensor(parameters[i][1])
                i = i + 1

    def print_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)


def split_dataset(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y.astype('int'), test_size=test_size)
    return X_train, X_test, y_train, y_test


def read_table(args):
    logging.info("Reading in decision table ´%s´...", args.table)

    df = pd.read_csv(args.table)
    X_train, X_test, y_train, y_test = \
        split_dataset(df.values[:, :5], df.values[:, 5].astype('int'))

    logging.info("Instantiating dataloaders for train and testset.")
    logging.info("Training batch size: %s, Test batch size: %s",
                 args.batch_size, args.batch_size)

    train_loader = get_dataset_loader(
        X_train, y_train, batch_size=args.batch_size)
    test_loader = get_dataset_loader(
        X_test, y_test, batch_size=args.batch_size)
    return train_loader, test_loader


def get_dataset_loader(X_np, y_np, batch_size=128, shuffle=True):
    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).long()
    dataset = torch.utils.data.TensorDataset(X, y)
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle)
    return dataset_loader


def read_table(args):
    logging.info("Reading in decision table ´%s´...", args.table)

    df = pd.read_csv(args.table)
    X_train, X_test, y_train, y_test = \
        split_dataset(df.values[:, :5], df.values[:, 5].astype('int'))

    logging.info("Instantiating dataloaders for train and testset.")
    logging.info("Training batch size: %s, Test batch size: %s",
                 args.batch_size, args.batch_size)

    train_loader = get_dataset_loader(
        X_train, y_train, batch_size=args.batch_size)
    test_loader = get_dataset_loader(
        X_test, y_test, batch_size=args.batch_size)
    return train_loader, test_loader


def load_dataset(csv_file, num_classes):
    dataset = genfromtxt(csv_file, delimiter=',')
    X = dataset[:,0:5]
    y = dataset[:,5]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return X_train, X_test, y_train, y_test

