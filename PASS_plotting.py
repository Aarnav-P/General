# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 08:32:47 2022

@author: ME

This code just imports and displays basic data
"""

import numpy as np
import matplotlib.pyplot as plt
import os

filepath = os.path.dirname(__file__)
data = np.genfromtxt(filepath + "\\poly_data.csv")


x = data[:,0]
y = data[:,1]

# n = input("What degree polynomial would you like: ")
n = 4
poly = np.polyfit(x,y,n)
# print(poly)

# y2 = poly[0]*(x**(n)) + poly[1]*(x**(n-1)) + poly[2]*(x**(n-2)) + poly[3](x**(n-3)) + poly[4]
y2 = (0)

for i in range(n+1):
    y2 = y2 + poly[i]*(x**(n-i))
    
    

error = data[:,2]

plt.errorbar(x, y, yerr = error)
plt.xlabel("x")
plt.ylabel("y=f(x)")
plt.plot(x,y2)
plt.show()
