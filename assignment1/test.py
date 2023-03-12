#!/usr/bin/env python
""" Developed as part of Assignment 1 of COM3240.
The code tests Neural Network of varying size.
It can be used as perceptron or as multi-layer feed-forward network
depending on the invoked methods. 
The code has been based on the material from Lab1 and Lab2 
of COM3240 Adaptive Intelligence created by Dr Matthew Ellis.
 """

import csv
import math
import matplotlib.pyplot as plt
import numpy as np;
import numpy.matlib 
from scipy.io import loadmat
import neural_network

__author__ = "Aleksandra Kulbaka"
__credits__ = ["Dr Matthew Ellis", "Grant Sanderson", "Michael Nielsen", "Michal Daniel Dobrzanski"]
__version__ = "1.0.1"
__email__ = "awkulbaka1@sheffield.ac.uk"

# read EMNIST train and test datasets
# divide the input in train and test sets by 255 to normalize data so each pixel has value from 0 to 1
emnist = loadmat('emnist-letters-1k')
x_train = emnist['train_images'] / 255
train_labels = emnist['train_labels']
x_test = emnist['test_images'] /255
test_labels = emnist['test_labels']

# dataset contains letter A - Z so 26 different labels
n_labels = 26
# create labels vectors. e.g., label 0 corresponds to a vector [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_train = np.zeros((train_labels.shape[0], n_labels))
y_test  = np.zeros((test_labels.shape[0], n_labels))
for i in range(train_labels.shape[0]):   
    y_train[i, train_labels[i].astype(int)] = 1
for i in range(test_labels.shape[0]):    
    y_test[i, test_labels[i].astype(int)] = 1

# global parameters constant during all the tests: number of epochs, size of batch, learning rate eta
N_EPOCH = 250
BATCH_SIZE = 50 
ETA = 0.05


# create list of tuples (pixels, desired output vector)
# train_data as list of tuples is useful for e.g. easy random split of data for training and validation
train_data = list(zip(x_train, y_train))
test_data = list(zip(x_test, y_test))

# parameters changed during experiments: lambda for L1, sizes of hidden layers, parameter indicating applying or skipping L1 regularisation
lambd = 0.003
hidden_1 = 100
hidden_2 = 100
if_L1 = True

# create NeuralNetwork object by giving arguments: layers' sizes, 
# tuples of train and test data, number of epochs, batch size, eta, 
# lambda and boolean value indicating applying or skipping L1 regularisation
net = neural_network.NeuralNetwork([784, hidden_1, hidden_2, 26], train_data, test_data, N_EPOCH, BATCH_SIZE, ETA, lambd, if_L1)
# call train_multilayer method with arguments: size of neural network, 
# test set (test_data or validate_data) and name of the test data
print(net.train_multilayer(4, net.test_data, "Test "))




# Below there is a code for all the experiments I've done in my report. Uncomment to make it work. 

# # TASK 3, 4: single layer perceptron with average weight update matrix, on a test data 
# if_L1 = False
# net = neural_network.NeuralNetwork([784, 26], train_data, test_data, N_EPOCH, BATCH_SIZE, ETA, lambd, if_L1)
# print(net.train_perceptron(net.test_data, "Test "))


# # TASK 6:
# hidden_1 = 50
# if_L1 = True
# # vary lambda
# lambd = 0.003

# # creates or add to the existing file the result of 5 runs of neural network 
# with open("task6.txt", "a") as f:
#     f.write("Lambda value = {}".format(lambd))
#     f.write("\n")
#     for i in range(5):
#       net = neural_network.NeuralNetwork([784, hidden_1, 26], train_data, test_data, N_EPOCH, BATCH_SIZE, ETA, lambd, if_L1)
#       result = net.train_multilayer(3, net.validate_data, "Validation ") + "\n"
#       f.write(result)
#       f.write("\n")
#       print(result)

# # TASK 7:
# lambd = 0.003
# if_L1 = True
# # vary hidden layer's size
# hidden_1 = 50

# # creates or add to the existing file the result of 10 runs of neural network 
# with open("task7.txt", "a") as f:
#     f.write("1st Hidden layer = {}".format(hidden_1))
#     f.write("\n")
#     for i in range(10):
#       net = neural_network.NeuralNetwork([784, hidden_1, 26], train_data, test_data, N_EPOCH, BATCH_SIZE, ETA, lambd, if_L1)
#       result = net.train_multilayer(3, net.test_data, "Test ") + "\n"
#       f.write(result)
#       f.write("\n")
#       print(result)

# # TASK 8:
# lambd = 0.003
# hidden_1 = 100
# if_L1 = True
# #vary second hidden layer's size
# hidden_2 = 100

# # creates or add to the existing file the result of 10 runs of neural network 
# with open("task8.txt", "a") as f:
#     f.write("2nd hidden layer = {}".format(hidden_2))
#     f.write("\n")
#     for i in range(10):
#       net = neural_network.NeuralNetwork([784, hidden_1, hidden_2, 26], train_data, test_data, N_EPOCH, BATCH_SIZE, ETA, lambd, if_L1)
#       result = net.train_multilayer(4, net.test_data, "Test ") + "\n"
#       f.write(result)
#       f.write("\n")
#       print(result)  
