#!/usr/bin/env python
""" Developed as part of Assignment 1 of COM3240.
The code implements Neural Network of varying size.
It can be used as perceptron or as multi-layer feed-forward network
depending on the invoked methods. 
The code has been based on the material from Lab1 and Lab2 
of COM3240 Adaptive Intelligence created by Dr Matthew Ellis.
2 methods has been inspired by the code described by Michael Nielsen 
in the book "Neural Networks and Deep Learning" 
and wrote in Python3 by Michal Daniel Dobrzanski.
 """

import math
import matplotlib.pyplot as plt
import numpy as np;
import numpy.matlib 
import random

__author__ = "Aleksandra Kulbaka"
__credits__ = ["Dr Matthew Ellis", "Grant Sanderson", "Michael Nielsen", "Michal Daniel Dobrzanski"]
__version__ = "1.0.1"
__email__ = "awkulbaka1@sheffield.ac.uk"

class NeuralNetwork(object):

    def __init__(self, sizes, train_data, test_data, n_epoch, batch_size, eta, lambd, regularisation = False):
        """sizes - contains the number of neurons in the respective layers of the network, 
        train and test data - lists of tuples of letters in form: (pixels, desired vector output), 
        number of epochs, batch size, training rate eta, lambda,
        boolean value indicating applying or skipping L1 regularisation."""

        self.sizes = sizes
        self.train_data = train_data
        self.test_data = test_data
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.eta = eta
        self.lambd = lambd
        self.regularisation = regularisation

        # generate random sets of weights and biases
        self.biases = [np.random.randn(y,) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(1/x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        # shuffle train data and take 20% of it as a validate set
        random.shuffle(self.train_data)
        self.validate_data = self.train_data[:5200]
        self.train_data = self.train_data[5200:]

        self.n_train_samples = len(self.train_data)
        self.n_batches = int(math.ceil(self.n_train_samples/batch_size))

        # initialize lists to store training accuracy and error for each epoch
        self.errors = np.zeros((n_epoch,))
        self.accuracy = np.zeros((n_epoch,))
        
        # upzip train_data into array of pixels and array of desired vector outputs 
        self.inputs_train, self.labels_train = zip(*self.train_data)


    def train_multilayer(self, layers, test, test_name):
        """ layers - number of layers in the neural network,
        test - set of data to test the accuracy: validate_data or test_data,
        test_name - name of the validation set"""

        for epoch in range(self.n_epoch):
            # divide the training data into 784 x 50 matrices representing mini batches 
            mini_batches = [self.inputs_train[b : b + self.batch_size] for b in range(0, self.n_train_samples, self.batch_size)]
            desired_labels = [self.labels_train[b : b + self.batch_size] for b in range(0, self.n_train_samples, self.batch_size)]

            # x0: matrix of 50 initial inputs in the batch
            for index, x0 in enumerate(mini_batches):
                x0 = np.transpose(x0)

                # initialise matrices for storing biases and weights updates computed by the backpropagation
                biases_update = [np.zeros(b.shape) for b in self.biases]
                weights_update = [np.zeros(w.shape) for w in self.weights]

                # x1: input layer -> hidden layer,
                h1 = np.transpose(np.transpose(np.matmul(self.weights[0], x0)) + self.biases[0])
                x1 = relu(h1)
                
                # x2: hidden layer -> output layer or hidden layer1 -> hidden layer2 (for 2 hidden layers)
                h2 = np.transpose(np.transpose(np.matmul(self.weights[1], x1)) + self.biases[1])
                x2 = relu(h2)  

                if layers == 4:
                    # hidden layer2 -> output layer
                    h3 = np.transpose(np.transpose(np.matmul(self.weights[2], x2)) + self.biases[2])
                    x3 = relu(h3)              

                    error_signal = np.transpose(desired_labels[index]) - x3
                else:
                    error_signal = np.transpose(desired_labels[index]) - x2 

                if self.regularisation:
                    # compute the L1 regularisation penalty
                    L1_error = [self.lambd * np.sum(np.sqrt(np.square(weight))) / self.n_train_samples for weight in self.weights]
                    L1_error = np.sum(L1_error)

                    self.errors[epoch] = self.errors[epoch] + (0.5 * np.sum(np.square(error_signal)) / self.n_train_samples) + L1_error
                else:
                    self.errors[epoch] = self.errors[epoch] + 0.5 * np.sum(np.square(error_signal)) / self.n_train_samples   

                # backpropagation
                if layers == 4:
                    # output layer -> hidden layer2
                    delta3 = relu_dev(h3) * error_signal
                    weights_update[2] =  np.matmul(delta3, np.transpose(x2))
                    biases_update[2] = np.sum(delta3, axis=1)

                    # hidden layer2 -> hidden layer1
                    delta2 = relu_dev(h2) * np.matmul(self.weights[2].T, delta3)
                    weights_update[1] = np.matmul(delta2, np.transpose(x1))
                    biases_update[1] = np.sum(delta2,axis=1)

                else:        
                    # output layer -> hidden layer1
                    delta2 = relu_dev(h2) * error_signal
                    weights_update[1] = np.matmul(delta2, np.transpose(x1))
                    biases_update[1] = np.sum(delta2,axis=1)

                # hidden layer -> input layer
                delta1 = relu_dev(h1) * np.matmul(self.weights[1].T, delta2)
                weights_update[0] = np.matmul(delta1, np.transpose(x0))
                biases_update[0] = np.sum(delta1,axis=1)

                # updating weights and biases
                if self.regularisation:
                    self.weights = [weight + self.eta / self.batch_size * (update - self.lambd * signum(weight)) 
                    for (weight, update) in zip(self.weights, weights_update)]
                else:
                    self.weights = [weight + self.eta * update / self.batch_size 
                    for (weight, update) in zip(self.weights, weights_update)]

                self.biases = [bias + self.eta * update / self.batch_size 
                for (bias, update) in zip(self.biases, biases_update)]

            # get accuracy and error of the testing data
            accuracy, error = self.evaluate(test)
            print( "Epoch {}. On {} data: Error: {}, Accuracy: {}".format(epoch + 1, test_name, error, accuracy))
        return "{} set: Error: {}, Accuracy: {}".format(test_name, error, accuracy)            
    
            
    def train_perceptron(self, test, test_name):
        """ test - set of data to test the accuracy: validate_data or test_data,
        test_name - name of the validation set """
        
        # initialise average matrix and constant tau
        average_matrix = np.zeros((self.n_epoch,))
        deltas = []
        TAU = 0.01

        for epoch in range(self.n_epoch):
            # divide the training data into 784 x 50 matrices representing mini batches
            mini_batches = [self.inputs_train[b : b + self.batch_size] for b in range(0, self.n_train_samples, self.batch_size)]
            desired_labels = [self.labels_train[b : b + self.batch_size] for b in range(0, self.n_train_samples, self.batch_size)]
            # initialise deltas to store the sum of weights updates 
            deltas = [np.zeros(w.shape) for w in self.weights]

            # x0: matrix of 50 initial inputs in the batch
            for index, x0 in enumerate(mini_batches):
                x0 = np.transpose(x0)

                # initialise matrices for storing biases and weights updates computed by the backpropagation
                biases_update = [np.zeros(b.shape) for b in self.biases]
                weights_update = [np.zeros(w.shape) for w in self.weights]

                # x1: input layer -> output layer
                h1 = np.transpose(np.transpose(np.matmul(self.weights[0], x0)) + self.biases[0])
                x1 = relu(h1)            

                error_signal = np.transpose(desired_labels[index]) - x1

                if self.regularisation:
                    # compute the L1 regularisation penaltyy
                    L1_error = [self.lambd * np.sum(np.sqrt(np.square(weight))) / self.n_train_samples for weight in self.weights]
                    L1_error = np.sum(L1_error)

                    self.errors[epoch] = self.errors[epoch] + (0.5 * np.sum(np.square(error_signal)) / self.n_train_samples) + L1_error
                else:
                    self.errors[epoch] = self.errors[epoch] + 0.5 * np.sum(np.square(error_signal)) / self.n_train_samples   

                # backpropagation: output layer -> input layer
                delta1 = relu_dev(h1) * error_signal
                weights_update[0] = np.matmul(delta1, np.transpose(x0))
                biases_update[0] = np.sum(delta1,axis=1)

                # updating weights and biases
                if self.regularisation:
                    self.weights = [weight + self.eta / self.batch_size * (update - self.lambd * signum(weight)) 
                    for (weight, update) in zip(self.weights, weights_update)]
                else:
                    self.weights = [weight + self.eta * update / self.batch_size 
                    for (weight,update) in zip(self.weights, weights_update)]

                self.biases = [bias + self.eta * update / self.batch_size 
                for (bias, update) in zip(self.biases, biases_update)]

                deltas[0] += (self.eta * weights_update[0] / self.batch_size)
            
            # compute average matrix
            if epoch == 0:
                average_matrix[epoch] = np.sum(deltas[0]) / self.n_batches
            else:
                average_matrix[epoch] = average_matrix[epoch - 1] * (1 - TAU) + TAU * np.sum(deltas[0] / self.n_batches)

            accuracy, error = self.evaluate(test)
            print( "Epoch {}. Average Matrix: {}.  On {} data: Error: {}, Accuracy: {}".format(epoch + 1, 
            average_matrix[epoch], test_name, error, accuracy))
        
        # plot the average matrix value vs epochs
        plt.plot(average_matrix)
        plt.ylabel('Average Matrix')
        plt.xlabel('Epochs')
        plt.show()        
        return "{} set: Error: {}, Accuracy: {}".format(test_name, error, accuracy) 


# the code for the following 2 functions was inspired by the code from the book: 
# http://neuralnetworksanddeeplearning.com/
# and the following repository: 
# https://github.com/MichalDanielDobrzanski/DeepLearningPython
    def feed_forward(self, x):
        """Return the output of the network for x"""
        for b, w in zip(self.biases, self.weights):
            x = relu(np.dot(w, x) + b)            
        return x

    def evaluate(self, test):
        """ Take test dataset as an argument 
        return the accuracy and errors of neural network on this dataset """
        test_results = [(np.argmax(self.feed_forward(pixels)), np.argmax(label_vec)) for (pixels, label_vec) in test]
        accuracy = sum(int(output == label) for (output, label) in test_results) / len(test)

        errors = [(np.square(self.feed_forward(pixels) - label_vec)) for (pixels, label_vec) in test]
        errors = 0.5 * np.sum(errors) / len(test)

        return accuracy, errors 


def relu(x):
    """The relu function."""
    return np.maximum(0, x)
    # return np.where(x > 0, x, 0.01*x)

def relu_dev(x):
    """Derivative of the relu function. 
    Return 0.01 if x <= 0 instead of 0 to prevent the dying relu problem"""
    # https://cs231n.github.io/neural-networks-1/#actfun
    return np.where(x > 0, 1, 0.01)

def signum(x):
    """The signum function"""
    return np.where(x > 0, 1, -1)

    