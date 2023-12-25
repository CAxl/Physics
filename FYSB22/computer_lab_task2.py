import numpy as np
import matplotlib.pyplot as plt 

plt.rcParams['text.usetex'] = True

s = 4   # spatial separation of wells
xi_max = 8 + s
xi_min = -xi_max
N  = 100*2*xi_max    # num steps

xi = np.linspace(xi_min, xi_max, N) # discretized x-range
h = (xi_max - xi_min)/N     # step-size

v0 = 6

def V(xi):  # potential function doubble well
    return -v0*1/(np.cosh(xi + s)) - v0*1/(np.cosh(xi - s))

def f(xi, eps): # f(xi)
    return -v0 * ((np.cosh(xi + s)**(-2)) + (np.cosh(xi - s)**(-2))) - eps

def phi(eps):   # wave function, recursion relation
    phi = np.zeros(N)
    q = np.sqrt(-eps) 
    phi[0] = 1
    phi[1] = np.exp(q*h)
    for i in range(1, N-1):
        phi[i+1] = (2 + h**2 * f(xi[i], eps)) * phi[i] - phi[i-1]
    return phi

#-------------------
# Bisection method

def nodes(phi):
    nodes = 0
    for i in range(N-1):
        if np.sign(phi[i+1]) != np.sign(phi[i]):
            nodes += 1    
    return nodes

def bisec(desiredNodes, Tol = 10**(-8)):
    emin = -2*v0
    emax = -0.1 
    cond = True
    nodemin = nodes(phi(emin))
    nodemax = nodes(phi(emax))

    while cond:
        if nodemin > desiredNodes or nodemax < desiredNodes:    # give up
            raise ValueError("The desired solution was not found in the range specified")

        e = (emin + emax)/2     # bisect

        Phi = phi(e)
        if nodes(Phi) == desiredNodes:
            if np.sign(Phi[-2] * Phi[-1] - np.exp(np.sqrt(-e)*h) * Phi[-1]**2) == -1:
                emin = e
            else:
                emax = e
        if nodes(phi(e)) < desiredNodes:
            emin = e
        elif nodes(phi(e)) > desiredNodes:
            emax = e
        
        if np.abs((emin - emax) / (emin + emax)) < Tol: # if close to zero, end while
            cond = False
            
    return e


epsilon0 = bisec(0) # 0 nodes --> even parity (grst)
epsilon1 = bisec(1) # 1 node --> odd parity (grst)
epsilon2 = bisec(2) # 2 node --> even parity (1st excited state)
epsilon3 = bisec(3) # 3 node --> odd parity (1st excited state)

#---------------

def norm(e):    # returns normalization constant
    f = np.square(phi(e))  
    S = 0
    for i in f:
        S += i*h
    return 1/(np.sqrt(S))  
 
#--------------
# plotting

plt.plot(xi, norm(epsilon0)*phi(epsilon0)+epsilon0, label="even parity",color="tab:blue")
plt.plot(xi, norm(epsilon1)*phi(epsilon1)+epsilon1, label="odd parity",color="tab:orange")
plt.plot(xi, norm(epsilon2)*phi(epsilon2)+epsilon2,color="tab:blue")
plt.plot(xi, norm(epsilon3)*phi(epsilon3)+epsilon3,color="tab:orange")
plt.plot(xi, V(xi), label=r"potential function V($\xi$)", color="black")
plt.ylabel(r"Effective energy, $\epsilon$")
plt.xlabel(r"Effective position, $\xi$")
plt.tick_params(direction = "in", right = "True", top = "True")
plt.legend()
plt.title("Eigen-states in a double potential well")
plt.text(8.3, -3.6, r"n = 1")
plt.text(8.3, -0.6, r"n = 2")
plt.show()
