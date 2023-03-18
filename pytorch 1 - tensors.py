# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 00:04:12 2023

@author: User
"""
# Why pytorch and not tensorflow?
# pytorch is less automated, I am aiming to get a fuller understanding of how using machine learning actually works in practice, so that I can independently use it for any project I decide to apply it to.

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

#test of correct installation
# =============================================================================
# x = torch.rand(5, 3)
# print(x)
# =============================================================================

# NUMPY VS TORCH

# =============================================================================
# # SIMILAR OPERATIONS
# n = np.linspace(0,1,5)
# t = torch.linspace(0,1,5)
# 
# n2 = np.arange(48).reshape(3,4,4)
# t2 = torch.arange(48).reshape(3,4,4)
# 
# print(n)
# print(t)
# print(n2)
# print(t2)
# # reshapes an array of nat ints 0-47 and shapes them into 3 4x4 matrices, 
# # aka a 3,4,4 tensor
# =============================================================================

# what is broadcasting?
# =============================================================================
# # numpy operates on two arrays if they are equal, or one array is equal to 1
# # therefore, multiplication (for example) is element-wise
# # i.e.
#
# a = np.ones((1,6,4), dtype = float)
# b = np.ones((5,6,1))
#
# # are compatible, each dimension of the array can be multiplied with its corresponding dimension
# # else: operands could not be broadcast together with shapes ...
# # NB: multiplication starts from the rightmost dimension
# 
# c = np.ones((6,5))
# d = np.arange(5)     # shape of d is (1,5)
# 
# print(c+d)
#
# # i.e. operating on an array dimension order 1 with order n stretches the order 1 array n times 
# 
# # similarly:
#
# e = torch.ones((6,5))
# f = torch.arange(5)
# print((e+f)-(c+d))
#
# # note that the main difference is the 'tensor' instead of 'array' and the different dtype
# 
# # if higher dimensions are not specified, they will be broadcast as ones
#
# colours = torch.randn((256,256,3)) #version of .rand that instead draws from a standard normal dist.
# scale = np.array([1.5,2,0.5])
# Tscale = torch.from_numpy(scale) # tool to convert
# Result = colours*Tscale
# print(Result)
#
# # note that scale is a 1D array-->tensor. Since broadcasting is right-to-left, "empty" dimensions are filled as 1
# 
# # Say you have two images
#
# pics = torch.randint(0,256,(2,256,256,3))
# scales = torch.tensor([1.5,2,0.5,0.5,2,1.5]).reshape(2,1,1,3) # by our previous rules, this now broadcasts with pics
# print(scales)
# =============================================================================

# n.b. trying to mix arrays and tensors will give a concatenation error or something similar, due to different dtypes

#OPERATIONS ACROSS DIMENSIONS

# =============================================================================
# #some basic operations
# t = torch.tensor([0.5,1,4,6])
# print(torch.mean(t), torch.std(t), torch.max(t), torch.min(t))
# 
# # what if I want to find means in a multidimensional array?
# # I have to specify the axis (direction) I'm taking means along.
# # e.g. to take means for the first element in each row (mean of a column), the axis is 0
# 
# nums = torch.arange(20, dtype= float).reshape(5,4) #arange defaults dtype to torch.int64, for some reason doesn't like not using float in multidim?
# print(torch.mean(nums, axis=0))
# 
# images = torch.randn((5,256,256,3))
# # say we want the mean of the R value across all 256 pixels over 5 images
# print(torch.mean(images,axis=0).shape) # axis = 0 specifies we're taking a mean across the first axis, 5.
# # we expect a grid of 256,256 with an average for R,G, and B, in each grid square
# # i.e. shape is (256,256,3)
# 
# # say we want the mean of the RGB channel as a whole 
# # we expect 5 grids of 256x256 with a number in each grid square for the average RGB value
# # i.e. shape is (5,256,256), we are taking a mean across the last axis, 3
# print(torch.mean(images, axis=-1).shape)
# 
# # similarly, axis 1 would be finding the average across the rows, which is to say the average of each column
# # and axis = -2 would be finding the average in each row , or 'across' the columns.
# # these would have the same shape
#  
# # finally, lets suppose we want to sort the pixels to find the brightest value of R, G, or B in each cell
# values, indices = (torch.max(images, axis = -1))
# print(values.shape) # the actual value of the highest R,G, or B, value per cell
# print(indices.shape) # the index of the cell that was highest (i.e. 0 for R, 1 if it was G, ...)
# 
# # revisit these?
# # =============================================================================
# # print(torch.std(nums, axis=0))
# # print()
# # print(torch.max(nums, axis=-0)) # note that it returns 
# # print()
# # print(torch.min(nums, axis=0))
# # =============================================================================
# =============================================================================

# WHY PYTORCH OVER NUMPY

# --> Automatic computation of gradients

x = torch.tensor([[3,4],[5,6]], dtype = float, requires_grad=True) # note only float and complex data can req gradients
y = (x**3).sum()
y.backward() # computes gradient (STORED IN X ITSELF)
# print(x.grad) # this returns 3(x_i)**2 as expected

# this is relevant because x will store the weights of every element 
# for machine learning, this is a good way to automatically keep track of how some small changes affects other tensors

# --> Pytorch is faster (~2x on CPU, a LOT more on GPU)
vals = torch.zeros(1)
for i in range(10):
    
    A = torch.randn(1000,1000)
    B = torch.randn(1000,1000)
    
    t1 = time.perf_counter()
    torch.matmul(A,B)
    t2 = time.perf_counter()
    # print('pytorch takes',t2-t1,'seconds.')
    
    a = np.random.randn(int(1e6)).reshape(1000,1000)
    b = np.random.randn(int(1e6)).reshape(1000,1000)
    
    T1 = time.perf_counter()
    a@b
    T2 = time.perf_counter()
    # print('numpy takes',T2-T1,'seconds.')
    sample = torch.tensor([(T2-T1)/(t2-t1)])
    vals = torch.cat((vals, sample))
    i = i+1
    #print('pytorch is',(T2-T1)-(t2-t1),'seconds faster, or',str((T2-T1)/(t2-t1))+'x faster')

print(torch.mean(vals))

# not exactly the fastest computation but the mean value I got from averaging 1000 results was that pytorch was 2.0182x quicker











