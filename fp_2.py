# Author: Daniel Chang
# Date: 3/24/2023
# Purpose: Using efficient cancer data "read_training_data" function we are using QR
# algorithm to find the linear model of the training data. Using this linear model we compare
# with other data to determine whether or not the prediction is accurate or not.

import numpy as np
from numpy import linalg as LA
from sympy import *
from efficient_cancer_data import read_training_data

fileNameA = "train.data"
fileNameB = "validate.data"
def QR(A):
    Q, R = np.linalg.qr(A, 'reduced')
    return Q, R

def leastSquares(A, b):
    # Given the fact that read_training_data() already gives us A and b
    # It is random everytime we rerun the program if D is not set.
    # I assume we need to calculate to only get the means to get a leastsquares model
    Q, R = QR(A)
    Rx = QRleastSquares(Q, R, b)
    # Row reducing and separating the matrix and the uneccessary numbers
    xhat = backSub(R, Rx)
    return xhat

def QRleastSquares(Q, R, b):
    # Rx = Q^Tb
    Qb = Q.transpose()*b 
    return Qb

def backSub(A, b):
    n = A.shape[0]
    A = np.array(A, float)
    x = np.zeros(n)
    x[-1] = b[-1] / A[-1,-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - np.dot(A[i,i+1:], x[i+1:])) / A[i,i]
    return x

def linModel(xhat, Avalidate):
    row, col= np.shape(Avalidate)
    x = np.zeros(row)
    # converting to an np array to manipulate the row more easily
    Avalidate = np.array(Avalidate)
    for i in range(row):
        calc = xhat.dot(Avalidate[i, :])
        # calc = 0 
        # for j in range(col):
        #     calc += xhat[j]*Avalidate[i,j]
        if calc > 0:
            x[i] = 1
        else:
            x[i] = -1
    return x
    
def percentageWrong(model, bval):
    count = 0
    n = len(model)
    # messed up bval being in a row instead of a column, tranposing to make it easier to compare columns rather than row and column.
    bval = bval.transpose()
    for i in range(n):
        if bval[i] == model[i]:
            count += 1
    percentWrong = (n-count)/n
    return percentWrong

def main():
    A, b = read_training_data(fileNameA)
    xhatTrain = leastSquares(A, b)
    Avalidate, bval = read_training_data(fileNameB)
    validateBM = linModel(xhatTrain, Avalidate)
    # Printing the linear model for part A
    print(xhatTrain)
    
    validatePercent = percentageWrong(validateBM, bval)
    trainingBM = linModel(xhatTrain,A)
    trainPercent = percentageWrong(trainingBM, b)
    # Print statement for part b.
    # ValidatePercent is 0.03076923076923077 = 3.07% inaccurate 
    print(validatePercent)
    # Print statement for part c
    print(trainPercent)
    # trainPercent is 0.04666666666666667 = 4.667% inaccurate 
    # If we look at the graph of the linear model, we will see more cases below the curve,
    # this could also mean that the training data had points that were much larger above the curve but fewer points.
    # This may have caused the inaccuracy of the training data to be higher than the actual validating data.
    return
