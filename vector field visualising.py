# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 19:06:28 2023

@author: User
"""

# exploring how to plot vector fields

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#2D

#set a range for axes
xrange = np.linspace(-10,10, 20)
yrange = np.linspace(-10,10, 20)

# create a grid of points
x,y = np.meshgrid(xrange, yrange) 

#set x component and y component of vector field
fx = np.sin(x)
fy = np.exp(np.cos(x))

# the quiver library seem to allow for much better graphing capability
font = {   'family' : 'monospace', 
           'weight' : 'normal',
           'size'   : 14,}
           
lines = {'lw' : 1,
         'ls' : '-',}

mpl.rc('font', **font) # changes formatting of mplb
mpl.rc('lines', **lines)

plt.figure(figsize=(6,6)) # scaling axes a bit to improve legibility
plt.quiver(x,y,fx,fy, pivot='middle') # centres arrows based off of midpoint as opposed to base
plt.axis("scaled")
plt.show()

