import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy 
from sympy.physics.wigner import wigner_3j

N = 1000
L = 10

r = np.linspace(0.01, L, N)
dr = np.abs(r[1]-r[0])

print("dr = ", dr)

def V_WS(r):
    """
    Woods-Saxon potential 16-O
    """
    R0 = 3.13
    a = 0.67
    V0 = 53.2

    mc2 = 939.57
    hc = 197.326
    const = 2*mc2 / (hc**2) # 2m/hbar^2 constant infront of V(r) in radial SE
    
    V = -V0/(1+np.exp((r-R0)/a))

    return const * V


def SE_solver(V,l):
    """
    Solves the radial SE numerically,
    Takes as input a potential function, V, and oribital angular momentum l.
    Returns (eigvals, eigvecs)
    """
    
    tridiag = - 1/(dr**2) * ( np.diag(np.full(N, -2)) + np.diag(np.ones(N-1),k=1) + np.diag(np.ones(N-1),k=-1))
    V_diag = np.diag(V)
    L_diag = np.diag(np.ones(N) * l*(l+1)/(r**2))
    
    H = tridiag + V_diag + L_diag   # Hamiltonian 
    
    eigvals, eigvecs = np.linalg.eig(H)
    
    sorted_i = np.argsort(eigvals)  # sort eigvals and eigvecs
    eigvals = eigvals[sorted_i]
    eigvecs = eigvecs[:, sorted_i]
    eigvecs = eigvecs.T 
    
    return eigvals, eigvecs

phi_Op_j = SE_solver(V_WS(r),1)[1][0] / r   # Radial eigenfunction (WS-potential) with n=0,l=1
R_fi_quadpole = scipy.integrate.simpson(phi_Op_j**2 * r**4)  # R_fi^(lamb) integral

B = (R_fi_quadpole)**2 / (4*np.pi)
print("B(E2) = ", B, "e^2fm^4")



# Wigner symbols

W = wigner_3j(1/2, 3/2, 2, 1/2, -1/2, 0)
print(W)
