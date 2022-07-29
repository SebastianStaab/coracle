# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:20:40 2022
@author: Sebastian Staab

Implementation according to tutorial by Johannes S. Fischer:
https://johfischer.com/2021/08/06/correlation-based-feature-selection-in-python-from-scratch/

Build to work on numpy arrays instead of pandas dataframes, 
only use continuous features and combine it all in a usable function
"""


import numpy as np
from math import sqrt


def getMerit(x, y):
    """
    function that computes the Merit

    Parameters
    ----------
    x : numpy array, 
            continuous feature(s)
    y : numpy array, 
            continuous target variable 

    Returns
    -------
    float
            Merit value

    """
    n,k = x.shape #n= sample size, k = number of feature subsets

    # average feature-target correlation
    rcf_all = []
    for feature in range(k):
        corr = np.corrcoef(y = y.flatten(), x = x[:,feature] )
        corr[np.tril_indices_from(corr)] = np.nan
        rcf_all.append( np.nanmean(abs( corr ) ))
    rcf = np.mean( rcf_all )
    
    # average feature-feature correlation
    corrff = np.corrcoef(x.transpose())
    corrff[np.tril_indices_from(corrff)] = np.nan
    corrff = abs(corrff)
    rff = np.nanmean(corrff)
    
    #merit calculation
    return (k * rcf) / sqrt(k + k * (k-1) * rff)



# Priority Queue, data structure to store and handle priorities
class PriorityQueue:
    def  __init__(self):
        #initialize empty queue
        self.queue = []

    def isEmpty(self):
        #returns 0 if queue is empty
        return len(self.queue) == 0
    
    def push(self, item, priority):
        """
        item already in priority queue with smaller priority:
        -> update its priority
        item already in priority queue with higher priority:
        -> do nothing
        if item not in priority queue:
        -> push it
        """
        for index, (i, p) in enumerate(self.queue):
            if (set(i) == set(item)):
                if (p >= priority):
                    break
                del self.queue[index]
                self.queue.append( (item, priority) )
                break
        else:
            self.queue.append( (item, priority) )
        
    def pop(self):
        #returns item with highest priority and remove it from queue
        max_idx = 0
        for index, (i, p) in enumerate(self.queue):
            if (self.queue[max_idx][1] < p):
                max_idx = index
        (item, priority) = self.queue[max_idx]
        del self.queue[max_idx]
        return (item, priority)


def cfs(x, y): 
    """
    correlation based feature selection (Hall 2000)
    best first search with max 5 backtracks. 
    Parameters
    ----------
    x : numpy array, 
            shape (n [#samples], k [#features]) input data (features)
    y : numpy array, 
            shape (n [#samples], input data (target variable)


    Returns
    -------
    best subset: numpy array, 
            index of chosen features

    """
    n,k = x.shape
    # initialize queue
    queue = PriorityQueue()
    
    #search for the first (best) feature to start a feature subset based on the merit
    best_value = -1
    best_feature = 0
    for feature in range(k):
        corr = np.corrcoef(y = y.flatten(), x = x[:,feature] )
        corr[np.tril_indices_from(corr)] = np.nan
        abs_coeff = np.nanmean(abs(corr))
        if abs_coeff > best_value:
            best_value = abs_coeff
            best_feature = feature
    
    
    # push first tuple (subset, merit)
    queue.push([best_feature], best_value)
    # list for visited nodes
    visited = []
    # counter for backtracks
    n_backtrack = 0
    # limit of backtracks
    max_backtrack = 5    
    
    
    # repeat until queue is empty
    # or the maximum number of backtracks is reached
    while not queue.isEmpty():
        # get element of queue with highest merit
        subset, priority = queue.pop()
        
        # check whether the priority of this subset
        # is higher than the current best subset
        if (priority < best_value):
            n_backtrack += 1
            
        else:
            best_value = priority
            best_subset = subset
            
    
        # goal condition
        if (n_backtrack == max_backtrack):
            break
        
        # iterate through all features and look of one can
        # increase the merit
        for feature in range(k):
            temp_subset = subset + [feature]
            
            # check if this subset has already been evaluated
            for node in visited:
                if (set(node) == set(temp_subset)):  
                    break
                # if not, ...
            else:
                # ... mark it as visited
                visited.append( temp_subset )
                
                # ... compute merit
                merit = getMerit(x[:, temp_subset], y)
                # and push it to the queue
                queue.push(temp_subset, merit)
         
    return np.array(best_subset) #returns index of best feature subset