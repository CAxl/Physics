import numpy as np
import matplotlib.pyplot as plt 

#plt.rcParams['text.usetex'] = True

xi_max = 8
xi_min = -xi_max
N  = 100*2*xi_max    # steps

xi = np.linspace(xi_min, xi_max, N) # discretized x-range
h = (xi_max - xi_min)/N     # step-size

# epsilon0 = -3.999018577   # epsilon_0
# epsilon1 = -0.9983       # epsilon_1

n = 0 # wave number
v0 = 6

def f(xi, eps):
    return -v0 * np.cosh(xi)**(-2) - eps

def V(xi):
    return -v0*1/(np.cosh(xi))

def phi(eps):
    phi = np.zeros(N)
    q = np.sqrt(-eps) 
    phi[0] = 1
    phi[1] = np.exp(q*h)
    for i in range(1, N-1):
        phi[i+1] = (2 + h**2 * f(xi[i], eps)) * phi[i] - phi[i-1]
    return phi

#------------------------------------------
#plotting using shooting method

# fig, (ax1, ax2) = plt.subplots(2,1)   
# ax1.plot(xi, phi(epsilon0))
# ax1.set_xlabel(r"effective position, $\xi$")
# ax1.set_ylabel(r"$\phi_0(\xi)$")
# ax1.tick_params(direction = "in", right = "True", top = "True")
# ax2.plot(xi, phi(epsilon1))
# ax2.set_xlabel(r"effective position, $\xi$")
# ax2.set_ylabel(r"$\phi_1(\xi)$")
# ax2.tick_params(direction = "in", right = "True", top = "True")
# plt.show()

#------------------------------------------
# bisection method

def nodes(phi):
    nodes = 0
    for i in range(N-1):
        if np.sign(phi[i+1]) != np.sign(phi[i]):
            nodes += 1
        
    return nodes

def bisec(desiredNodes, Tol = 10**(-8)):
    emin = -v0
    emax = -0.1 
    cond = True
    nodemin = nodes(phi(emin))
    nodemax = nodes(phi(emax))

    while cond:
        if nodemin > desiredNodes or nodemax < desiredNodes:    # give up
            raise ValueError("The desired solution was not found in the range specified")

        e = (emin + emax)/2

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
        
        if np.abs((emin - emax) / (emin + emax)) < Tol:
            cond = False
            
    return e

epsilon0 = bisec(0)
epsilon1 = bisec(1)


def norm(e):    # returns normalization constant
    f = np.square(phi(e))  
    S = 0
    for i in f:
        S += i*h
    return 1/(np.sqrt(S))   

plt.plot(xi, norm(epsilon0)*phi(epsilon0)+epsilon0, label="ground state",color="tab:blue")
plt.plot(xi, norm(epsilon1)*phi(epsilon1)+epsilon1, label="first excited state",color="tab:orange")
plt.plot(xi, V(xi), label="potential function V(x)", color="black")
plt.tick_params(direction = "in", right = "True", top = "True")
plt.legend()
plt.show()

       



