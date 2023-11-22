# -*- coding: utf-8 -*-
"""
Created on Oct 26 11:40:29 2023

@author: Aarnav Panda

This code numerically computes an n-order fourier series of some unknown function
between 0 and +2pi
"""
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy import signal


def n_order_cos(x, n):
    return np.cos(n*x)

def n_order_sin(x,n):
    return np.sin(n*x)


def fourier_coeffecients(func, n):
    
    
    constant = integrate.quad(func,-np.pi,np.pi)[0]/(2*np.pi)
    A_coeffecients = np.array([])
    B_coeffecients = np.array([])
    for order in range(n):
        
        A_n = integrate.quad(lambda x: func(x)*n_order_sin(x, order+1), -np.pi, np.pi)[0]/np.pi
        
        A_coeffecients = np.append(A_coeffecients, A_n)
        
        B_n = integrate.quad(lambda x: func(x)*n_order_cos(x, order+1), -np.pi, np.pi)[0]/np.pi
        B_coeffecients = np.append(B_coeffecients, B_n)

        
    return(constant, A_coeffecients, B_coeffecients)


def plot_functions(func, n):
    x = np.linspace(np.pi,-np.pi,1000)
    y = func(x)
    plt.xlim(-5,5)
    plt.plot(x, y, label = "function")
    
    A0, An, Bn = fourier_coeffecients(func, n)

    print("A0: ", A0)
    print("sin coeffecients: ", An)
    print("cos coeffecients: ", Bn)
    
    series = np.full(1000,A0)
    for order in range(n):
        series += An[order]*n_order_sin(x, order+1)
        series += Bn[order]*n_order_cos(x, order+1)
        
    plt.plot(x,series, label = "series n={}".format(n))
    plt.legend()
    

"""
Make the below function anything you want, change the order of the series as well.
"""
def function(x):
   # x = x % (2*np.pi) # This ensures whatever function is inputted has the right period
    return abs(x)

plot_functions(function, 11)
"""
i recommend trying a couble of special functions such as sawtooth and square wave

Also try to look at the resulting coeffecients of a simple cosine function

You can also remove the modulo operator from the function and see what happens
"""







