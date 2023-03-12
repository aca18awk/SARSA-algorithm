#!/usr/bin/env python
""" Developed as part of Assignment 2 of COM3240.
The code defines the experiments specified in the assignment brief.
To run any of them, invoke the desired function.

The code has been based on the material from Lab8 and Lab9 
of COM3240 Adaptive Intelligence created by Dr Matthew Ellis.
 """

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import homing_task as network

__author__ = "Aleksandra Kulbaka"
__credits__ = ["Dr Matthew Ellis"]
__version__ = "1.0.1"
__email__ = "awkulbaka1@sheffield.ac.uk"


def task2(draw_plot = True):
    ''' run of basic implementation of SARSA
        draw_plot: whether to draw learning curve at the end '''
    n_repetitions = 15         # number of runs for the algorithm
    n_trials = 1000            # number of trials
    n_steps = 50               # maximum number of allowed steps
    learning_rate = 0.5        # learning rate, [0, 1]
    epsilon = 0.1              # exploration rate, [0, 1]
    gamma = 0.9                # discount faction, [0, 1)
    lambd = 0.6                # lamda for eligibility trace, [0, 1]

    eligibility_trace = False  # specifies whether to implement eligibility trace
    with_walls = False         # specifies whether to add walls to the world
    fixed_end = False          # specifies whether to set end point at [0, 2]
        
    learning_curve = np.zeros((n_repetitions, n_trials))
    optimal_path = np.zeros((n_repetitions, n_trials))
    weights = []

    for i in range(n_repetitions):
        print("Iteration ", i + 1)
        learning_curve[i], optimal_path[i], weight = network.robot_navigation(
            n_trials, n_steps, learning_rate, epsilon, gamma, 
            with_walls, fixed_end, eligibility_trace, lambd)
        weights.append(weight)  
    
    if draw_plot:
        network.plot_learning_curve(learning_curve, n_repetitions, n_trials)
    return learning_curve

def task3():
    ''' Find the optimum values of the parameters '''
    # example test 
    n_repetitions = 30
    n_trials = 1000  
    n_steps = 50        
    learning_rate = 0.7    # tested values: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]
    epsilon = 0.0          # tested values: [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    gamma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1]
    
    lambd = 0.6            # tested values: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    eligibility_trace = False
    with_walls = False
    fixed_end = True
        
    for g in gamma:
        print("For gamma: ", g, " average distance to the optimum is: ")
        for i in range(n_repetitions):
            learning_curve, optimal_path, weight = network.robot_navigation(
                n_trials, n_steps, learning_rate, epsilon, g, 
                with_walls, fixed_end, eligibility_trace, lambd)

            avg_distance = (learning_curve - optimal_path)
            print(np.mean(avg_distance))


def task4():
    ''' Compare eligibility trace in simple world with implementation from task 2'''
    n_repetitions = 15
    n_trials = 1000
    n_steps = 50 
    learning_rate = 0.5 
    epsilon = 0.1
    gamma = 0.9     
    lambd = 0.7      
    eligibility_trace = True
    with_walls = False
    fixed_end = False
        
    learning_curve = np.zeros((n_repetitions, n_trials))
    optimal_path = np.zeros((n_repetitions, n_trials))
    weights = []

    for i in range(n_repetitions):
        print("Iteration ", i + 1)
        learning_curve[i], optimal_path[i], weight = network.robot_navigation(
            n_trials, n_steps, learning_rate, epsilon, gamma, 
            with_walls, fixed_end, eligibility_trace, lambd)
        weights.append(weight)  

    learning_curve2 = task2(False)
    network.plot_learning_curve(learning_curve, n_repetitions, n_trials, learning_curve2)


def task4_2():
    ''' Add eligibility trace in more complex world '''
    n_repetitions = 15
    n_trials = 1000
    n_steps = 200
    learning_rate = 0.7
    epsilon = 0.0
    gamma = 0.99
    lambd = 0.6 
    eligibility_trace = False
    with_walls = True
    fixed_end = True
        
    learning_curve = np.zeros((n_repetitions, n_trials))
    optimal_path = np.zeros((n_repetitions, n_trials))
    weights = []

    for i in range(n_repetitions):
        print("Iteration ", i + 1)
        learning_curve[i], optimal_path[i], weight = network.robot_navigation(
            n_trials, n_steps, learning_rate, epsilon, gamma, 
            with_walls, fixed_end, eligibility_trace, lambd)
        weights.append(weight)  

    eligibility_trace = True
    learning_curve2 = np.zeros((n_repetitions, n_trials))
    optimal_path2 = np.zeros((n_repetitions, n_trials))
    weights2 = []

    for i in range(n_repetitions):
        print("Iteration ", i + 1)
        learning_curve2[i], optimal_path2[i], weight = network.robot_navigation(
            n_trials, n_steps, learning_rate, epsilon, gamma, 
            with_walls, fixed_end, eligibility_trace, lambd)
        weights2.append(weight) 

    network.plot_learning_curve(learning_curve, n_repetitions, n_trials, learning_curve2)

def task5():
    ''' Plot the information about the preferred direction in 2 different ways '''
    n_repetitions = 15
    n_trials = 1000
    n_steps = 200
    learning_rate = 0.7
    epsilon = 0.0
    gamma = 0.99
    lambd = 0.6 
    eligibility_trace = False
    with_walls = False
    fixed_end = True
        
    learning_curve = np.zeros((n_repetitions, n_trials))
    optimal_path = np.zeros((n_repetitions, n_trials))
    weights = []

    for i in range(n_repetitions):
        print("Iteration ", i + 1)
        learning_curve[i], optimal_path[i], weight = network.robot_navigation(
            n_trials, n_steps, learning_rate, epsilon, gamma, 
            with_walls, fixed_end, eligibility_trace, lambd)
        weights.append(weight)  

    network.plot_single_direction(weights, with_walls)
    network.plot_all_directions(weights, with_walls)


def task7():
    ''' Add walls to the world '''
    n_repetitions = 15
    n_trials = 1000
    n_steps = 200
    learning_rate = 0.7
    epsilon = 0.0
    gamma = 0.99
    lambd = 0.6 
    eligibility_trace = False
    with_walls = True
    fixed_end = True
        
    learning_curve = np.zeros((n_repetitions, n_trials))
    optimal_path = np.zeros((n_repetitions, n_trials))
    weights = []

    for i in range(n_repetitions):
        print("Iteration ", i + 1)
        learning_curve[i], optimal_path[i], weight = network.robot_navigation(
            n_trials, n_steps, learning_rate, epsilon, gamma, 
            with_walls, fixed_end, eligibility_trace, lambd)
        weights.append(weight)  

    network.plot_single_direction(weights, with_walls)
    network.plot_all_directions(weights, with_walls)


# task2()
# task3()
# task4()
# task4_2()
# task5()
task7()