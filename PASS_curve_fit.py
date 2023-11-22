# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:29:54 2022

@author: olive
Updated: Aarnav P
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
import random 
from scipy.optimize import curve_fit

filepath = os.path.dirname(__file__)
data = np.genfromtxt(filepath + "\\lorentzian_data.csv", skip_header=1)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font) # changes formatting of mplb

# below are the things you need to pass to curve fit
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
# cov returns a covariance matrix with values along the leading diagonal

fig = plt.figure(dpi=300)
axis = fig.add_subplot(111)
axis.errorbar(x, y, yerr = error) #plt is general, define fig then use axis will allow you to make multiple figures
axis.set_xlabel("x")
axis.set_ylabel("y=f(x)")

axis.plot(x, func(x, *fit))

print("Fitted variables are:")
variable = ["a", "b", "c"]
for i in range(3):
    print("{0}={1:.3f}+-{2:.3f}".format(variable[i],fit[i], np.sqrt(cov[i,i])))
