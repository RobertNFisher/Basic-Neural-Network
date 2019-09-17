# Course: CS4242
# Student name: Robert Fisher
# Student ID: 000380203
# Assignment: Assignment 3
# Due Date: 4/26/2019
# Signature: Robert Neil Fisher


import random as rnd
from math import exp
import numpy as np

class precept_neural_network:
    def __init__(self):
        self.learning_rate = .1
        self.bias = 1
        # Creates an input layer that takes in each node as a value of 0 or 1 adding a bias of 1
        self.inputs = [0,0,0,0,self.bias]
        # Creates a layer for outputs
        self.output_layer = [0]
        # Creates weighted values and randomly gives it a value
        self.weights = [0,0,0,0,0]

        for index in range(0, self.weights.__len__()):
            self.weights[index] = rnd.random() - .5

    def train(self, train_data):
        # Feed Forward
        index = 0
        for input in train_data[0]:
            self.inputs[index] = input
            index += 1

        # input layer -> output layer
        for j in range(0, self.output_layer.__len__()):
            for i in range(0, self.inputs.__len__()):
                self.output_layer[j] = self.weighted_sum(weights=self.weights, previous_layer=self.inputs)

        if self.output_layer[0] > 0:
            self.output_layer[0] = 1
        else:
            self.output_layer[0] = -1


        # Backpropigate
        for index in range(0, self.weights.__len__()):
            self.weights[index] += self.learning_rate * (train_data[1] - self.output_layer[0]) * self.inputs[index]

    def test(self, test_data):
        # Feed Forward
        index = 0
        for input in test_data[0]:
            self.inputs[index] = input
            index += 1

        # input layer -> output layer
        for j in range(0, self.output_layer.__len__()):
            for i in range(0, self.inputs.__len__()):
                self.output_layer[j] = self.weighted_sum(weights=self.weights, previous_layer=self.inputs)

        if self.output_layer[0] > 0:
            self.output_layer[0] = 1
        else:
            self.output_layer[0] = -1

        accuracy = 0
        if self.output_layer[0] == test_data[1]:
            accuracy = 1

        return accuracy


    '''
        Helper Classes
    '''


    # returns the weighted sums of all the values from the previous layer nodes
    def weighted_sum(self, weights, previous_layer):
        wsum = 0
        i = 0
        for weight in weights:
            wsum += weight * previous_layer[i]
            i += 1

        return wsum

    def loss_function(self, output, expected_output, input):
        self.learning_rate * (expected_output - output) * input


class multi_layer_network:
    def __init__(self):
        self.learning_rate = .1
        self.bias = 1
        self.hidden_nodes = 3
        # Creates an input layer that takes in each node as a value of 0 or 1 adding a bias of 1
        self.inputs = [0,0,0,0,self.bias]
        # Creates a hidden layer
        self.hidden_layer = np.zeros(self.hidden_nodes)
        # Creates a layer for outputs
        self.output_layer = [0, 0]
        # Creates weights between input layer and hidden layer with values randomly assigned between -0.5 and 0.5
        self.input_hidden_weights = np.zeros((self.hidden_nodes, self.inputs.__len__()))

        for col in range(0, self.input_hidden_weights.__len__()):
            for row in range(0, self.input_hidden_weights[col].__len__()):
                self.input_hidden_weights[col][row] = rnd.random() - .5

        # Creates weights between hidden and output layer
        self.hidden_output_weights = np.zeros((self.output_layer.__len__(), self.hidden_nodes))

        for col in range(0, self.hidden_output_weights.__len__()):
            for row in range(0, self.hidden_output_weights[col].__len__()):
                self.hidden_output_weights[col][row] = rnd.random() - .5

    def train(self, train_data):
        # Feed Forward
        index = 0
        for input in train_data[0]:
            self.inputs[index] = input
            index += 1

        # input layer -> hidden layer
        for j in range(0, self.hidden_layer.__len__()):
            for i in range(0, self.inputs.__len__()):
                self.hidden_layer[j] = self.weighted_sum(weights=self.input_hidden_weights[j], previous_layer=self.inputs)
                # activate
                self.hidden_layer[j] = self.sigmoid(self.hidden_layer[j])


        # hidden layer -> output layer
        for j in range(0, self.output_layer.__len__()):
            for i in range(0, self.hidden_layer.__len__()):
                self.output_layer[j] = self.weighted_sum(weights=self.hidden_output_weights[j], previous_layer=self.hidden_layer)
                # activate
                self.output_layer[j] = self.sigmoid(self.output_layer[j])

        # adjust output layer
        for index in range(0, self.output_layer.__len__()):
            self.output_layer[index] = self.output_layer[index] * (1 - self.output_layer[index]) * \
                                       (train_data[2][index] - self.output_layer[index])

        new_hidden_layer = np.zeros(self.hidden_layer.__len__())
        # Calculate the derived hidden layer
        reversed_wsum = 0
        for i in range(0, self.hidden_layer.__len__()):
            for j in range(0, self.output_layer.__len__()):
                reversed_wsum += self.hidden_output_weights[j][i] * self.output_layer[j]
            new_hidden_layer[i] = reversed_wsum * self.hidden_layer[i] * (1 - self.hidden_layer[i])

        # adjust the weights with the new values
        for i in range(0, self.hidden_layer.__len__()):
            for j in range(0, self.output_layer.__len__()):
                self.hidden_output_weights[j][i] += self.hidden_layer[i] * self.learning_rate

        # apply changes to the hidden layer
        for index in range(0, self.hidden_layer.__len__()):
            self.hidden_layer[index] = new_hidden_layer[index]

        # Calculate the input layer derivative
        for i in range(0, self.inputs.__len__()):
            for j in range(0, self.hidden_layer.__len__()):
                self.input_hidden_weights[j][i] += self.learning_rate * self.hidden_layer[j] * self.inputs[i]

    def test(self, test_data):
        # Feed Forward
        index = 0
        for input in test_data[0]:
            self.inputs[index] = input
            index += 1

        # input layer -> hidden layer
        for j in range(0, self.hidden_layer.__len__()):
            for i in range(0, self.inputs.__len__()):
                self.hidden_layer[j] = self.weighted_sum(weights=self.input_hidden_weights[j],
                                                         previous_layer=self.inputs)
                # activate
                self.hidden_layer[j] = self.sigmoid(self.hidden_layer[j])

        # hidden layer -> output layer
        for j in range(0, self.output_layer.__len__()):
            for i in range(0, self.hidden_layer.__len__()):
                self.output_layer[j] = self.weighted_sum(weights=self.hidden_output_weights[j],
                                                         previous_layer=self.hidden_layer)
                # activate
                self.output_layer[j] = self.sigmoid(self.output_layer[j])

        # calculate the accuracy of the current example
        accuracy = 0
        for index in range(0, self.output_layer.__len__()):
            accuracy += (test_data[2][index] - self.output_layer[index])**2

        return accuracy


    # returns the weighted sums of all the values from the previous layer nodes
    def weighted_sum(self, weights, previous_layer):
        wsum = 0
        i = 0
        for weight in weights:
            wsum += weight * previous_layer[i]
            i += 1

        return wsum

    def loss_function(self, output, expected_output, input):
        self.learning_rate * (expected_output - output) * input

    # Squishes value into a number between 0 and 1
    def sigmoid(self, value):
        return 1 / (1 + exp(-value))