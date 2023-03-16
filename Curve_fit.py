# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:29:54 2022

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import random 
from scipy.optimize import curve_fit

# os.path is a non-specific means of using any systems path infrastructure

filepath = os.path.dirname(__file__) 
#__file__ is a variable that contains the path to the module that is currently being imported. 
    # In this case, since __file__ is not preceded by a . to make it an extension,
    # it simply returns the path of the current file

# filepath = os.getcwd() # this is an alternative that just 
data = np.genfromtxt(filepath + "\\lorentzian_data.csv")

# aesthetic change parameters, creating dicts to pass to mpl.rc
font = {'family' : 'monospace',
           'weight' : 'normal',
           'size'   : 10,}
           
lines = {'lw' : 1,
         'ls' : '-',}

mpl.rc('font', **font) # changes formatting of mplb
mpl.rd('lines', **lines)

# define a function to fit to: 
def func(x, a, b, c):
    return a*x**2 / ((x-b)**2 + c)

x = data[:,0]
y = data[:,1]
error = data[:,2]

a = random.randint(0,10)
b = random.randint(0,10)
c = random.randint(0,10)

guess = [a,b,c]


fit, cov = curve_fit(func, x, y, p0=guess, sigma=error, absolute_sigma=True)
# cov returns a covariance matrix with values along the leading diagonal being the variances

# plt is general, define fig then use axis will allow you to make multiple figures
fig = plt.figure(dpi=300)
axis = fig.add_subplot(111)
axis.grid()
axis.errorbar(x, y, yerr = error,) 
axis.set_xlabel("x")
axis.set_ylabel("y=f(x)")

axis.plot(x, func(x, *fit))

print("Fitted variables are:")
variable = ["a", "b", "c"]
for i in range(3):
    print("{0}={1:.3f}+-{2:.3f}".format(variable[i],fit[i], np.sqrt(cov[i,i])))
