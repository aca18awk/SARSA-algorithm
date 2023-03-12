#!/usr/bin/env python
""" Developed as part of Assignment 2 of COM3240.
The code implements SARSA RL algorithm using machine learning. 
This file also contains methods to plot the learning curve 
and to plot the preferred direction of the robot. 

The code has been based on the material from Lab 8 and Lab 9 
of COM3240 Adaptive Intelligence created by Dr Matthew Ellis.
 """

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

__author__ = "Aleksandra Kulbaka"
__credits__ = ["Dr Matthew Ellis"]
__version__ = "1.0.1"
__email__ = "awkulbaka1@sheffield.ac.uk"

def robot_navigation(n_trials, n_steps, learning_rate, epsilon, gamma, with_walls, fixed_end, eligibility_trace, lambd=0.0):
    """ Main method implementing SARSA on Neural Network.
        n_trials: number of trials in one run,
        n_steps: maximum number of steps allowed,
        learning_rate: learning rate for SARSA,
        epsilon: exploration factor for epsilon-greedy, 
        gamma: discount factor for SARSA, 
        with_walls: boolean variable stating whether there are walls in the world,
        fixed_end: boolean variable to fix the end point to be [0, 2],
        eligibility_trace: boolean variable to implement eligibility trace,
        lambd: lambda for eligibility trace 

    
    """
    ## Definition of the environment
    N = 10                               # ---> number of rows
    M = 10                               # ---> number of columns
    N_STATES = N * M

    STATES_MATRIX = np.eye(N_STATES)
    N_ACTIONS = 4                                           #number of possible actions in each state: 1->N 2->E 3->S 4->W
    ACTION_ROW_CHANGE = np.array([-1,  0, +1,  0])          #number of cell shifted in vertical as a function of the action
    ACTION_COL_CHANGE = np.array([ 0, +1,  0, -1])          #number of cell shifted in horizontal as a function of the action
    
    # creates a list of allowed moves for each state
    ALLOWED_MOVES = {}
    for i in range(N):
        for j in range(M):
            moves = []
            if i > 0:
                moves.append(0)
            if i < N-1:
                moves.append(2)
            if j > 0:
                moves.append(3)
            if j < M-1:
                moves.append(1)
            moves.sort()
            ALLOWED_MOVES[str([i, j])] = moves

    # defining the end state
    if (fixed_end or with_walls):
        end = np.array([2, 0])
    else: 
        end = np.array([np.random.randint(N), np.random.randint(M)]) 
    s_end = np.ravel_multi_index(end, dims=(N, M), order = 'F')


    if eligibility_trace:
        elig = np.zeros((N_ACTIONS, N_STATES))

    learning_curve = np.zeros((n_trials))
    optimal_path = np.zeros((1, n_trials))
    weights = np.zeros((N_ACTIONS, N_STATES))   # how likely you are to go N, E, S, W from a given state
    rewards_list = np.zeros((N, M))             # reward received at each state

    # positive reward only when the robot reaches the end state
    rewards_list[end[0], end[1]] = 1     

    # negative reward when robot enters the walls                         
    if with_walls:
        for i in range(0, 5):
            rewards_list[3, i] = -0.1
        for i in range(6, 10):
            rewards_list[5, i] = -0.1
        for i in range(7, 10):
            rewards_list[i, 3] = -0.1

    max_steps_penalty = -1

    for trial in range(n_trials):

        #random start
        start = np.array([np.random.randint(N), np.random.randint(M)])   
        s_start = np.ravel_multi_index(start, dims=(N, M), order='F') 
        state = start     
        s_index = s_start
        step = 0

        # computing the optimal path from a given start point
        optimal = abs(start[0] - end[0]) + abs(start[1] - end[1])
        optimal_path[0, trial] = optimal


        while s_index != s_end and step <= n_steps:

            step += 1
            learning_curve[trial] = step
            input_vector = STATES_MATRIX[:, s_index].reshape(N_STATES, 1)
            Q_value = 1 / ( 1 + np.exp( - weights.dot(input_vector)))    # Qvalue = logsig(weights*input) 

            # allow to choose only the valid actions
            actions = ALLOWED_MOVES[str([state[0], state[1]])]
            greedy = (np.random.rand() > epsilon)
            if greedy:
                action = actions[np.argmax(Q_value[actions])]    #pick best action
            else:
                action = np.random.choice(actions)               #pick random action


            state_new = np.array([0, 0])
            state_new[0] = state[0] + ACTION_ROW_CHANGE[action]
            state_new[1] = state[1] + ACTION_COL_CHANGE[action]
                
            reward_current = rewards_list[state_new[0], state_new[1]]

            # if robot goes to the wall, return to the previous position
            if reward_current == -0.1:
                state_new = state
                
            s_index_new = np.ravel_multi_index(state_new, dims=(N, M), order='F')

            #store variables for sarsa computation in the next step
            output = np.zeros((N_ACTIONS, 1))
            output[action] = 1

            # update weights       
            if step > 1:
                delta = (reward_old - Q_value_old + gamma * Q_value[action]) 
                old_state = output_old.dot(input_old.T)    # previous (state, action)
                current_state = output.dot(input_vector.T) # current (state, action)
                if eligibility_trace:
                    elig += current_state        # add 1 to chosen action in a given state
                    dw = learning_rate * delta * elig * old_state 
                    weights += dw
                    elig = gamma * lambd * elig  # update all trace values
                else: 
                    dw = learning_rate * delta * old_state
                    weights += dw
                        

            #update variables for next step
            input_old = input_vector
            output_old = output
            Q_value_old = Q_value[action]
            reward_old = reward_current
            state[0] = state_new[0]
            state[1] = state_new[1]
            s_index = s_index_new

            if not reward_current == 0:
                dw = learning_rate * (reward_current - Q_value_old) * output_old.dot(input_old.T)
                weights += dw      
            
            # negative reward if no. of steps has been exceeded
            elif step == n_steps:
                dw = learning_rate * (max_steps_penalty - Q_value_old) * output_old.dot(input_old.T)
                weights += dw    

    return learning_curve, optimal_path, weights 



def plot_learning_curve(learning_curve, n_repetitions, n_trials, learning_curve2 = []):  
    ''' Given learning curve(s), number of runs of algorithm and number of trials,
        plots the learning curves with the error bars '''

    avg = np.mean( learning_curve, axis = 0)
    avg_errors = np.std(learning_curve, axis = 0) / np.sqrt(n_repetitions)
    smooth_avg_means = gaussian_filter1d(avg, 2)
    smooth_avg_errors = gaussian_filter1d(avg_errors, 2)

    plt.errorbar(np.arange(n_trials), smooth_avg_means, smooth_avg_errors,
     0, elinewidth = 0.1, capsize = 1, alpha = 0.2, color = "green")
    plt.plot(smooth_avg_means, 'tab:green')

    # if specified, plot second learning curve
    if not learning_curve2 == []:
        avg_2 = np.mean( learning_curve2, axis = 0)
        avg_errors_2 = np.std(learning_curve2, axis = 0) / np.sqrt(n_repetitions)
        smooth_avg_means_2 = gaussian_filter1d(avg_2, 2)
        smooth_avg_errors_2 = gaussian_filter1d(avg_errors_2, 2)

        plt.errorbar(np.arange(n_trials), smooth_avg_means_2, smooth_avg_errors_2,
         0, elinewidth = 0.1, capsize = 1, alpha = 0.2, color = "firebrick")
        plt.plot(smooth_avg_means_2, 'tab:red')

    plt.xlabel('Trial',fontsize = 16)
    plt.ylabel('Steps',fontsize = 14)
    plt.tick_params(axis = 'both', which='major', labelsize = 14)
    
    # uncomment to make y axis be from -0.1 to 50
    # plt.axis((-(n_trials/10.0), n_trials, -0.1, 50))
    plt.show() 


def plot_single_direction(weights, with_walls):
    ''' Given the weights from N runs (N >= 0),
        it plots the preferred action in each state
        and draws walls on the map if with_walls is True. '''

    weights = np.mean(weights, axis = 0)
    weights_1 = weights.T
    directions = np.zeros((100, 2))
    # find the action with the highest weight - the preferred action
    for i, weight in enumerate(weights_1):
        direc =  np.argmax(weight)
        if direc == 0:
            directions[i] = [-1, 0]
        elif direc == 1:
            directions[i] = [0, 1]
        elif direc == 2:
            directions[i] = [1, 0]
        else:
            directions[i] = [0, -1]

    U = np.reshape(directions[:, 0], (10, 10))
    V = np.reshape(directions[:, 1], (10, 10))
    quiver_graph(U, V, with_walls)


def plot_all_directions(weights, with_walls):
    ''' Given the weights from N runs (N >= 0),
        it plots the preferred directions in each state
        and draws walls on the map if with_walls is True. '''

    weights = np.mean(weights, axis = 0)
    weights_1 = weights.T
    directions = np.zeros((100, 2))
    for i, weight in enumerate(weights_1):
        directions[i][0] = weight[2] - weight[0]
        directions[i][1] = weight[1] - weight[3]

    U = np.reshape(directions[:, 0], (10, 10))
    V = np.reshape(directions[:, 1], (10, 10))
    U = U / np.sqrt(np.square(U) + np.square(V))
    V = V / np.sqrt(np.square(U) + np.square(V))

    quiver_graph(U, V, with_walls)


def quiver_graph(U, V, with_walls):
    """ given the arrows directions, plots them on the 10 x 10 grid """
    x = np.arange(1, 11, 1)
    y = np.arange(1, 11, 1)
    X, Y = np.meshgrid(x,y)
    # remove arrows from the walls and the end states
    if with_walls:
        for i in range(0, 5):
            U[i, 3] = 0
            V[i, 3] = 0
        for i in range(6, 10):
            U[i, 5] = 0
            V[i,5] = 0
        for i in range(7, 10):
            U[3, i] = 0
            V[3, i] = 0

    U[0, 2] = 0
    V[0, 2] = 0

    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, U, V)
    ax.quiverkey(q, X = 0.3, Y = 1.1, U = 10,
                label='Quiver key, length = 10', labelpos='E')

    # draw walls on the plot
    if with_walls:
        for i in range(0, 5):
            plt.plot(4, i+1, 's', markersize = 20, color="firebrick")
        for i in range(6, 10):
            plt.plot(6, i+1, 's', markersize = 20, color="firebrick")
        for i in range(7, 10):
            plt.plot(i+1, 4, 's', markersize = 20, color="firebrick")

    # draw end state on the plot
    plt.plot(3, 1, 'X', color = "limegreen", markersize = 20)
    plt.show()