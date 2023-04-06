# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:05:56 2023

@author: User
"""

import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt

# the idea here is generally comparable to curve fitting
# instead of variables however, we want to consider items that contain multiple bits of information
# hence, let's consider an independent vector x_1, and a dependent vector y_1

# what we want is a function f(x_i;a) = \hat{y_i}
# s.t. \hat{y_i} should be as close to the true vector, y_i as possible, by adjusting the parameters of the function, a.

# considering the relative importance and context of y_1, we can DESIGN a loss function L that tells us how 'far' our result \hat{y_i} is

# e.g. for a simple numerical distance L = \sum{y_i - \hat{y_i}}^2, or simply put a variance.

# example

x = torch.tensor([[6,2],[5,2],[1,3],[7,6]]).float()
y = torch.tensor([1,5,2,5]).float()

# there is an unknown rule here that takes in x_i and predicts y_i. Let's figure it out with ML.
# We're gonna give way more parameters than necessary (probably), but how?

# x is a 1 x 2 matrix. We multiply it by an 8 x 2 matrix to take in each set of 2 numbers and spit out an 8D vector
# we then multiply our 8x2 matrix by a 1x8 matrix to produce an output value.

# from our two defined matrices we have 24 parameters for a, whose initial values should be random.

M1 = nn.Linear(2,8,bias=False) # takes a 2d vector, returns 8d

# M1(x) applies M1 to each element x_i (also called an instance)

M2 = nn.Linear(8,1,bias=False)

y_hat = M2(M1(x)).squeeze() #we will actually get a [4,1] shape so squeeze removes the extra dimension of 1

# now we will train our network

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(2,8,bias=False)
        self.Matrix2 = nn.Linear(8,1,bias=False)
    def forward(self,x):
        x = self.Matrix1(x)
        x = self.Matrix2(x)
        return x.squeeze()
# by using the subclass nn.module we can call functions of that class, such as .parameters()
f = Net()
y_hat = f(x)
print(f(x))
for par in f.parameters():
    print(par)
    
L = nn.MSELoss() # takes variance, equiv to torch.mean(y,y_hat)

# THE MAIN IDEA is to find partial(L)/partial(a_i)
# s.t. a_i = a_i - l * delL/del(a_i), where l is the learning rate
# imagine L as a potential --> to seek the minimum of that curve by iteration we are effectively subtracting the gradient at a point.
# Like a very, very cautious version of the Newton-Raphson method, as l is usually quite small

# each pass over the full data set x is called an epoch. If the data set is too massive, the process might be done in many steps, adjusting in several dimensions each time.

opt = SGD(f.parameters(), lr=0.001) # SGD is stochastic gradient descent

losses = []
loss = L(f(x),y)
while loss > 0.1:
    opt.zero_grad() # reset from prev epoch
    loss = L(f(x),y) # compute loss
    loss.backward() # compute grad
    opt.step() # iterate using gradient
    losses.append(loss.item()) 

plt.plot(losses)



        






