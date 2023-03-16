# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:48:33 2022

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt

# Let's take the scenario of intensity varying by angle
# sample data is inputted below
Ierr = 0.05 
I = np.array([1.86,1.63,1.13,0.52,0.16,0.00,0.57,1.11,1.56,1.71])
Angledeg = np.linspace(0,180,10)
Anglerad = (np.pi/180)*Angledeg # converts to radians
Angle = np.cos(2*Anglerad) # this is not syntactically clean, but it makes the code parse a lot easier than replacing every instance of Angle with np.cos(2*Anglerad)

abar = np.sum(Angle)/len(Angle)
Ibar = np.sum(I)/len(I)
a2bar = np.sum((Angle**2)/len(Angle))
aI = np.sum(Angle*I)/len(I)           
# this calculates all necessary parameters for chi squared calc


m = (aI - abar*Ibar)/(a2bar - abar**2)
c = (Ibar*a2bar - aI*abar)/(a2bar - abar**2)
x = np.linspace(-1,1,20)

merr = np.sqrt((1/len(I)) * (1/(a2bar-abar**2)) * (Ierr**2))
cerr = np.sqrt((1/len(I)) * (a2bar) * (1/(a2bar-abar**2)) * (Ierr**2))

# For non linear functions you can just fiddle to make work, use parameters of Asin(wt + phi)
# however here we can plot I against cos(2theta) and approximate to
plt.xlabel('cos(2θ)')
plt.ylabel('Intensity')
plt.title('Plot of Intensity as a function of θ against cos(2θ)')
plt.errorbar(Angle,I,Ierr, fmt='gx',markersize=2) 
plt.plot(x,m*x+c)
plt.show()

# =============================================================================
# Chi2 = 0
# for i in range(10):
#     Chi2 = Chi2 + ((m*Angle[i] + c - I[i])/0.05)**2
# =============================================================================

Chi2 = np.sum(((m*Angle + c - I)/0.05)**2) # much better way to do this loop

print('Minimum value of the Goodness-of-Fit parameter Chi Squared is ',np.round(Chi2,3))
             
# print(f'The line of best fit has a gradient of {m} with an uncertainty of {merr} and an intercept of {c} with an uncertainty of {cerr}') # this shows off replacement fields
print('m =',np.round(m,3),'+-',np.round(merr,3))
print('c =',np.round(c,3),'+-',np.round(cerr,3))
# knowing that m - I1-I2 , c = I2
I1err =     np.sqrt(cerr**2 + merr**2)
print('I1 =', np.round((m+c),3),'+-',np.round(I1err,3))
print('I2 =',np.round(c,3),'+-',np.round(cerr,3))


