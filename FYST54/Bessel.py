import numpy as np
from scipy.integrate import odeint
from scipy.special import jn # bessel function
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import scipy


def fbessel(Y, x):
    nu = 1
    y = Y[0]
    z = Y[1]
  
    dydx = z
    dzdx = 1.0 / x**2 * ( - (x**2 - nu**2) * y)
    return [dydx, dzdx]

x0 = 0.01
y0 = 1
z0 = 0
Y0 = [y0, z0]

xspan = np.linspace(x0, 10, 1000)
sol = odeint(fbessel, Y0, xspan)


def normalize(func, x):
   norm = scipy.integrate.simpson(abs(func) ** 2, x)
   return func/np.sqrt(norm)



sol = normalize(sol[:,0],xspan)

plt.plot(xspan, sol, label='numerical soln')
plt.plot(xspan, jn(1, xspan), 'r--', label='Bessel')
plt.legend()
plt.show()
