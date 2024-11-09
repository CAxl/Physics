from cProfile import label
import scipy
from sympy import*
import numpy as np
import scipy 
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy import integrate
import matplotlib.pyplot as plt
from sklearn import preprocessing as pre


L = 20
r = np.linspace(0.01, L, 1000)
R0 = 5
V0 = 50

#solve odeint for chi
def oneparam(l):
    def dSdr(S,r):
        chi,v = S
        mu = 0.05
        k = 1    
        #V = np.piecewise(r, [r < R0, r>= R0], [-V0, 0])
        V = 0

        chi_r = v
        v_r = (l*(l+1)/r**2 + 2*mu*V - k**2)*chi

        S = [chi_r, v_r]

        return S
    return dSdr 


# plotting

for l in range(0,4):
    dSdr = oneparam(l)
    
    # initial conditions
    chi_0 = r[0]**l
    v_0 = r[0]**(l-1)
    S_0 = (chi_0, v_0)
    
    # get_sol chi
    sol = odeint(dSdr, S_0, t=r)
    chi = sol.T[0]/r
    
    chi = chi.reshape(-1,1)
    chi_scaled = pre.MinMaxScaler( (-1*np.abs(min(chi) / max(chi)),1) ).fit_transform(chi)  # rescale y-axis

    plt.plot(r, chi_scaled, label=f"R_{l}(r)")

V = np.piecewise(r, [r < R0, r>= R0], [-V0, 0], color="black", label="V(r)")
plt.plot(r, V/50)




plt.xlabel("r [fm]")
plt.legend()
plt.show()

