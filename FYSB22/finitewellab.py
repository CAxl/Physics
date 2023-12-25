# program for finding bound state energies


# import libraries
import numpy as np
import matplotlib.pyplot as plt

# define parameters
L = 4 * 10 ** (-9)  # width in meters
e = 1.6 * 10 ** (-19)  # electron charge
V0 = 0.22 * e  # potential well height, in Joule
Esteps = 1000  # number of steps in energy vector
E = np.linspace(0, V0, Esteps)  # energy vector
m0 = 9.1 * 10 ** (-31)  # free electron mass
me = 0.067 * m0  # effective electron mass
mh = 0.48 * m0  # effective hole mass
hbar = 1.05 * 10 ** (-34)  # plancks constant
np.seterr(divide="ignore")

# Define left and right hand side of (**)
RHS = np.sqrt((V0 - E) / E)  # right hand side
LHSe = np.tan(L / 2 * np.sqrt(2 * me / hbar ** 2 * E))
LHSh = np.tan(L / 2 * np.sqrt(2 * mh / hbar ** 2 * E))

# plot functions, x-axis in electron volts
plt.plot(E / e, RHS)
plt.plot(E / e, LHSe, label='LHS electrons')
plt.plot(E / e, LHSh, label='LHS holes')
plt.ylim([0, 1.5])  # set y-axis range
plt.xlim([0, 0.22])  # set x-axis range
plt.legend()
plt.show()
