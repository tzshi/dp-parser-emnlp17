#!/usr/bin/env python
# encoding: utf-8

from dynet import Saveable, parameter, transpose, vecInput, cmult, dropout
import numpy as np


class Dense(Saveable):
    def __init__(self, indim, outdim, activation, model):
        self.activation = activation
        self.W = model.add_parameters((outdim, indim))
        self.b = model.add_parameters(outdim)

    def __call__(self, x):
        return self.activation(parameter(self.W) * x + parameter(self.b))

    def get_components(self):
        return [self.W, self.b]

    def restore_components(self, components):
        self.W, self.b = components


class MultiLayerPerceptron(Saveable):
    def __init__(self, dims, activation, model):
        self.layers = []
        self.dropout = 0.
        self.outdim = []
        for indim, outdim in zip(dims, dims[1:]):
            self.layers.append(Dense(indim, outdim, activation, model))
            self.outdim.append(outdim)

    def __call__(self, x):
        for layer, dim in zip(self.layers, self.outdim):
            x = layer(x)
            if self.dropout > 0.:
                x = dropout(x, self.dropout)
        return x

    def set_dropout(self, droprate):
        self.dropout = droprate

    def get_components(self):
        return self.layers

    def restore_components(self, components):
        self.layers = components


class Bilinear(Saveable):
    def __init__(self, dim, model):
        self.U = model.add_parameters((dim, dim))

    def __call__(self, x, y):
        U = parameter(self.U)
        return transpose(x) * U * y

    def get_components(self):
        return [self.U]

    def restore_components(self, components):
        [self.U] = components


identity = lambda x: x

