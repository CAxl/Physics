from tkinter import NORMAL
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn # bessel function
from scipy.integrate import odeint
from scipy.integrate import simpson
import scipy


R0 = 5
V0 = 50



def V_WS(r):
    """
    Woods-Saxon potential 16-O
    """
    R0
    a = 0.67
    V0
    
    V = -V0/(1+np.exp((r-R0)/a))

    return V


def oneparam(l):
    def solver(Y, r):
        chi = Y[0]
        z = Y[1]
    
        mu = 0.1
        
        # potentials 
        # V = np.piecewise(r, [r < R0, r>= R0], [-V0, 0])
        V = V_WS(r)
        #V = 0
        k = 1

        dchi_dr = z
        dz_dr = chi*(l*(l+1)/r**2 + 2*mu*V - k**2)
    
        return [dchi_dr, dz_dr]
    return solver
    

def normalize(func, x): # normalize wavefunc
   norm = scipy.integrate.simpson(abs(func) ** 2 , x)
   return func/np.sqrt(norm)


# r-range
r0 = 1e-15
r_range = np.linspace(r0, 20, 1000)


for l in range(0,2):    # call solver for different l
    sols = oneparam(l)
    
    # initial conds
    chi0 = r_range[0]**l
    z0 = r_range[0]**(l-1)
    Y0 = [chi0, z0]

    sol = odeint(sols,Y0, r_range).T[0] # solve ivp
    sol = normalize(sol/r_range, r_range) # normalize solutions

    plt.plot(r_range, sol, label=f"R_{l}(r)")



V = np.piecewise(r_range, [r_range < R0, r_range>= R0], [-V0, 0])
#plt.plot(r_range, normalize(V, r_range), color="black", label="V(r)")
plt.plot(r_range, normalize(V_WS(r_range), r_range), color="black", label="V_WS(r)")
plt.xlabel("r [fm]")
plt.ylabel("R_l(r)")
plt.legend()
plt.show()


