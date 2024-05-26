import json
import random

import numpy as np   # check out how to install numpy
from utils import load, plot_sample
import math
import struct

# =========================================
#       Homework on K-Nearest Neighbors
# =========================================
# Course: Introduction to Information Theory
# Lecturer: Haim H. Permuter.
#
# NOTE:
# -----
# Please change the variable ID below to your ID number as a string.
# Please do it now and save this file before doing the assignment

ID = '315602284'

# Loading and plot a sample of the data
# ---------------------------------------
# The MNIST database contains in total of 60000 train images and 10000 test images
# of handwritten digits.
# This database is a benchmark for many classification algorithms, including neural networks.
# For further reading, visit http://yann.lecun.com/exdb/mnist/
#
# You will implement the KNN algorithm to classify two digits from the database: 3 and 5.
# First, we will load the data from .MAT file we have prepared for this assignment: MNIST_3_and_5.mat

Xtrain, Ytrain, Xvalid, Yvalid, Xtest = load('MNIST_3_and_5.mat')

# The data is divided into 2 pairs:
# (Xtrain, Ytrain) , (Xvalid, Yvalid)
# In addition, you have unlabeled test sample in Xtest.
#
# Each row of a X matrix is a sample (gray-scale picture) of dimension 784 = 28^2,
# and the digit (number) is in the corresponding row of the Y vector.
#
# To plot the digit, see attached function 'plot_sample.py'

sampleNum = 0
plot_sample(Xvalid[sampleNum, :], Yvalid[sampleNum, :])

# Build a KNN classifier based on what you've learned in class:
#
# 1. The classifier should be given a train dataset (Xtrain, Ytain),  and a test sample Xtest.
# 2. The classifier should return a class for each row in test sample in a column vector Ytest.
#
# Finally, your results will be saved into a <ID>.txt file, where each row is an element in Ytest.
#
# Note:
# ------
# For you conveniece (and ours), give you classifications in a 1902 x 1 vector named Ytest,
# and set the variable ID at the beginning of this file to your ID.

# < your code here >

Newimage = np.zeros(784)


def l1_dist(x1,x2):
    if x1[1].shape != x2[1].shape:
        raise Exception("the vectors dimensions are different")
    dist = np.zeros(len(x1))
    for i in range(len(x1)):
        dist[i] = math.fabs(x1[i] - x2[i])
    return dist

def l2_dist(x1,x2):
    if x1[1].shape != x2[1].shape:
        raise Exception("the vectors dimensions are different")

    dist = 0
    for i in range(len(x1)):
        dist = dist + math.pow(x1[i] - x2[i],2)
    dist = math.sqrt(dist)
    return dist

def predict_l1(x):
    db_size = len(Xtrain)
    distances = np.empty(db_size, dtype=np.ndarray)
    for i in range(db_size):
        distances[i] = l1_dist(Xtrain[i,:],x)
    norms = [] #Compute the norms of each row

    for j in range(len(distances)):
        norms.append(np.linalg.norm(distances[j]))
    k = 5
    indices = np.argsort(norms)[:k] # get the indices of the 5 vectors with the smallest norms
    results = []
    for i in range(len(indices)):
        results.append(Ytrain[indices[i]])
    if results.count(3) > results.count(5):
        prediction = 3
    else:
        prediction = 5
    return prediction

def valid_gd():
    successes = 0
    for i in range(len(Xvalid)):
        s1 = predict_l1(Xvalid[i]) #the prediction by l1 norm
        s2 = int(Yvalid[i]) #the answer
        print("Test number:",i+1)
        print("The prediction is",s1,"and the correct digit is",s2)
        if s1 == s2:
            successes = successes +1
            print(successes,"/",i+1)


    return

#valid_gd()

def test(Xtest):
    np.loadtxt('315602284.txt')
    successes = 0
    with open('315602284.txt','w')as f:
        for i in range(len(Xtest)):
            Ytest[i] = predict_l1(Xtest[i]) #the prediction by l1 norm
            f.write(str(Ytest[i])+'\n')
            print("Test number:",i+1)
            print("The prediction is",Ytest[i])
        np.savetxt(f'{ID}.txt', Ytest, delimiter=", ", fmt='%i')
        return Ytest


# Example submission array - comment/delete it before submitting:
Ytest = np.arange(0, Xtest.shape[0])

# save classification results
print('saving')
np.savetxt(f'{ID}.txt', Ytest, delimiter=", ", fmt='%i')
print('done')

test(Xtest)