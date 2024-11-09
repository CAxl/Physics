import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# discretize x-range

N = 1000
L = 10

r = np.linspace(0.01, L, N)
dr = np.abs(r[1]-r[0])

#dr = 0.01
#r0 * dr*1e-3

print("dr = ", dr)

def V(r):
    """
    Harmonic oscillator potential
    """
    E0 = -55    # MeV    
    hw = 8.6
    mc2 = 939.57
    hc = 197.326

    const = 2*mc2 / (hc**2) # const infront of V(r) in Radial SE

    V = ( 0.5 * (mc2)*(hw**2)* r**2 / (hc**2) ) + E0

    return const * V


def V_WS(r):
    """
    Woods-Saxon potential
    """
    R0 = 5.8
    a = 0.65
    mc2 = 939.57
    hc = 197.326
    const = 2*mc2 / (hc**2) # 2m/hbar^2 constant infront of V(r) in radial SE
    
    V = -50/(1+np.exp((r-R0)/a))

    return const * V

def V_SO(V,r,l):
       
    lamb = -0.5
    
    ls_plus = l
    ls_minus = -(l+1)
    
    dV = np.gradient(V)
    
    V_algn = lamb*(1/r)*dV*ls_plus
    V_anti = lamb*(1/r)*dV*ls_minus
    
    return V_algn, V_anti
    
    

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


# __________________________________________________________
# relevant constants
hw = 8.6
mc2 = 939.57
hc = 197.326
const = 2*mc2 / (hc**2)


# __________________ Analytical VS numerical solutions HO __________________
def E_nl(n,l):
    return hw*(2*n + l + (3/2)) - 55

print("Analytical energies HO:")
for n in range(3):
    for l in range(3):
        print(f"E_{n}{l} = ", E_nl(n,l))

print()
print("Numerical energies HO:")
for n in range(3):
    for l in range(3):
        print(f"E_{n}{l} = ", (hc**2)*SE_solver(V(r),l)[0][n] / (2*mc2))

print()

# print error
print("errors_nl = |anlytic_nl - numeric_nl|")
for n in range(3):
    for l in range(3):
        print(f"error_{n}{l} = ", np.abs(E_nl(n,l) - (hc**2)*SE_solver(V(r),l)[0][n] / (2*mc2)))
        
print()

# plotting HO solutions
ind = ["s","p","d","f","g"]
for n in range(3):
    for l in range(3):
        plt.plot(r, SE_solver(V(r),l)[1][n] / r , label=f"{n}{ind[l]}")
        
plt.legend()   
plt.xlabel("r [fm]")
plt.ylabel("R(r)")
plt.show()

#_____________________________________WS prints and plots______________________________________

# plot wavefunctions:
ind = ["s","p","d","f","g"]
for n in range(3):
    for l in range(3):
        plt.plot(r, SE_solver(V_WS(r),l)[1][n] / r  , label=f"{n}{ind[l]}")

plt.xlabel("r [fm]")
plt.ylabel("R(r)")
plt.legend()
plt.show()        

# print & plot eigenenergies 
print("Numerical solutions to WS potential")

for n in range(3):
    for l in range(3):
        E = (hc**2)*SE_solver(V_WS(r),l)[0][n] / (2*mc2)
        print(f"E_{n}{l} = ", E, "MeV") 
            
        values = np.abs(np.full(N,E) - V_WS(r)/const)
        ind = np.argmin(values)

        x = np.linspace(0,r[ind],N)
        
        plt.plot(x, np.full(N,E), label=f"E_{n}{l} = {np.round(E,2)}")
        
        
plt.plot(r,V_WS(r) / const, color="black", label="V_WS(r)") # plot potential function
plt.xlabel("r [fm]")
plt.ylabel("E [MeV]")
plt.legend()
plt.show()

# alternative plot u + \eps
ind = ["s","p","d","f","g"]
for n in range(3):
    for l in range(3):
        plt.plot(r, 200*SE_solver(V_WS(r),l)[1][n] / r + (hc**2)*SE_solver(V_WS(r),l)[0][n]/ (2*mc2),  label=f"{n}{ind[l]}")
        
plt.plot(r,V_WS(r) / const, color="black", label="V_WS(r)") # plot potential function
plt.xlabel("r [fm]")
plt.ylabel("E [MeV]")
plt.legend()
plt.show()





