import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random 
from scipy.optimize import curve_fit

data = np.genfromtxt("readings.txt", dtype='float', skip_header=1, delimiter=',')

Vs = data[:,0]
Vserror = 0.01*Vs + 10e-3
Vd = data[:,1]
stdevVd = data[:,2]
R = data[:,3]

Vr = Vs - Vd
I = Vr/R

Ierror = I*np.sqrt((np.sqrt(Vserror**2 + stdevVd**2)/(Vs-Vd))**2 + (0.01)**2)
lnIerror = Ierror/I

y = np.log(I)
x = Vd

#POLYFIT
# =============================================================================
# # n = input("What degree polynomial would you like: ")
# n = 1
# poly, cov = np.polyfit(x,y,n, cov=True)
# print(poly)
# fit_uncertainties = np.sqrt(np.diag(cov))
# print(np.sqrt(np.diag(cov)))
# # y2 = poly[0]*(x**(n)) + poly[1]*(x**(n-1)) + poly[2]*(x**(n-2)) + poly[3](x**(n-3)) + poly[4]
# y2 = (0)
# 
# for i in range(n+1):
#     y2 = y2 + poly[i]*(x**(n-i))
# =============================================================================

def func(x, a, b):
    return a*x + b

a = random.randint(-10,10)
b = random.randint(-10,10)

guess = [a,b]

fit, cov = curve_fit(func, x, y, p0=guess, sigma=lnIerror, absolute_sigma=True, maxfev=40000)

plt.plot(x, func(x, *fit)) # CURVEFIT
plt.errorbar(x, y, yerr=lnIerror, fmt='k.', markersize=3, label = 'Measured Uncertainties')
plt.ylabel('ln(I/1A)')
plt.xlabel('Vd')
# plt.plot(x,y2, 'r--', label='Linear Polyfit')  #POLYFIT
# plt.plot(a,b, label = 'extrapolated')
plt.legend()
plt.title('Plot of ln(I) against Diode Voltage. T=-1.2°C')
plt.savefig('boltzmann.png')
plt.show()

T = -1.2

print("Fitted variables are:")
variable = ["a", "b", "c"]
for i in range(2):
    print("{0}={1:.3f}+-{2:.3f}".format(variable[i],fit[i], np.sqrt(cov[i,i])))

#POLYFIT    
# print("The gradient is ", str('%.5g'%poly[0]),"±",str('%.5g'%fit_uncertainties[0]), 
#       "and the intercept is ", str('%.5g'%poly[1]),"±",str('%.5g'%fit_uncertainties[1]))
# denom = (1.6*(10**-19))/poly[0]

denom = (1.6*(10**-19))/fit[0]
print('kT =', '%.10g'%denom)
abs_zero = 297.3057742
sigma_abs_zero = 3.42950226
k = denom/(T+abs_zero)
sigma_k = k*np.sqrt((np.sqrt(cov[0,0])/fit[0])**2 + (sigma_abs_zero/abs_zero)**2)
print('k =', '%.10g'%k, '+-', '%.10g'%sigma_k)

# polyfit has an extra factor of 2 that can change things with smaller data sets.
# In the limit these converge to the same results as curvefit (apparently)

#print(x)
#print(y)
#print(lnIerror)
df = pd.DataFrame({"voltage" : x,"ln(I)" : y, "errors" : lnIerror})
df.to_csv('readingsLSFR.csv', index=False)