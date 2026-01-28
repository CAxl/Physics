import numpy as np
import matplotlib.pyplot as plt


def fermi(r, A): 
    R = 1.1 * A**(1/3)
    a = 0.57 #fm
    
    den = 1 + np.exp((r-R)/a)
    return 1 / den


rvals = np.linspace(0,10, 1000)

A1 = 10
A2 = 100
rho0 = 0.17 # fm

plt.plot(rvals, rho0*fermi(rvals, A1), label = f"A={A1}")
plt.plot(rvals, rho0*fermi(rvals, A2), label = f"A={A2}")
plt.xlabel("r [fm]")
plt.ylabel("e*rho(r) [e/fm^-3]")
plt.legend()
plt.show()


