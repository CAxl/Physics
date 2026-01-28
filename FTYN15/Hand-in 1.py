from math import sqrt
import numpy as np
import scipy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.integrate import quad


x = np.linspace(0,1000)

def f(x):
    return sqrt(x)/(np.exp(x)-1)

integral = scipy.integrate.quad(f,0,1000)
print(integral)
plt.plot(x,f(x))
plt.show()
