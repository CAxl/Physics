from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt


N = 1000
hbar = 1
omega = 1

n = np.diag(np.arange(N))

H = hbar * omega * (n + 0.5*np.eye(N))

print(H)

eigvecs = np.linalg.eigh(H)[1]
print(eigvecs) # bit vectors, one quanta in each (n:th) state


# spatial representation
# Hermite polynomials




# plot of ground state
m = 1
w = 1
hbar = 1

def V(x):
    
    V = (1/2)*m*pow(omega,2) * pow(xi,2)
    return V

xi = np.linspace(-2,2,1000)

plt.plot(xi, V(xi), color="black", linestyle="-.", label= "V(x)")

phi0 = np.exp(-xi**2/2)
plt.plot(xi, phi0, label = "phi0")

plt.legend()
plt.show()





