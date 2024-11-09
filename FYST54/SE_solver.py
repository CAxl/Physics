
import numpy as np
import matplotlib.pyplot as plt


# discretize x-range

N = 1000
L = 15

x = np.linspace(-L/2, L/2, N)
x = np.linspace(0.01, 10, N)
dx = np.abs(x[1]-x[0])
print(dx)


def tridiag():
    """
    define tridiagonal matrix which returns second derivative in x
    """

    return - 1/(dx**2) * ( np.diag(np.full(N, -2)) + np.diag(np.ones(N-1),k=1) + np.diag(np.ones(N-1),k=-1))

def L2(x,l):
    """
    L^2 matrix with (angular-eigval / r^2) as diagonal elements
    """
    return np.diag( np.ones(N) * l*(l+1)/(x**2))

def V(x):
    """
    Harmonic oscillator potential
    """
    E0 = -55    # MeV    
    hw = 8.6
    mc2 = 939.57
    hc = 197.326

    const = 2*mc2 / (hc**2)

    V = ( 0.5 * (mc2)*(hw**2)* x**2 / (hc**2) ) + E0

    return const * V

HO_diag = np.diag(V(x))

def V_WS(x):
    """
    Woods-Saxon potential
    """
    R0 = 5.8
    a = 0.65
    mc2 = 939.57
    hc = 197.326
    const = 2*mc2 / (hc**2) # 2m/hbar^2 constant infront of V(r) in radial SE
    
    V = -50/(1+np.exp((x-R0)/a))

    return const * V

WS_diag = np.diag(V_WS(x))


H_HO = HO_diag + tridiag() + L2(x,0)
H_WS = WS_diag + tridiag() + L2(x,0)

eigvals, eigvecs = np.linalg.eig(H_WS)

# sort eigvals and eigvecs
sorted_i = np.argsort(eigvals)
eigvals = eigvals[sorted_i]
eigvecs = eigvecs[:, sorted_i]
eigvecs = eigvecs.T

hw = 8.6
mc2 = 939.57
hc = 197.326

# for i in range(3):
#     print(f"E{i} = ", (hc**2)*eigvals[i] / (2*mc2) )
#     plt.plot(x, 5*eigvecs[i] + eigvals[i], label=f"u_n={i},l=0")
# plt.plot(x, V_WS(x), color="black", label="WS potential")
# plt.title("Woods Saxon")
# plt.legend()
# plt.show()


# print analytical
def E_nl(n,l):
    return 8.6*(2*n + l + (3/2)) - 55

# print()
# print("Analytical energies:")
# print("E00 = ", E_nl(0,0))
# print("E10 = ", E_nl(1,0))
# print("E20 = ", E_nl(2,0))
# print("E30 = ", E_nl(3,0))

# print("E10 = ", E_nl(1,0))
# print("E01 = ", E_nl(0,1))
# print("E11 = ", E_nl(1,1))
# print("E12 = ", E_nl(1,2))




def SE_solver(V,l):
    
    tridiag = - 1/(dx**2) * ( np.diag(np.full(N, -2)) + np.diag(np.ones(N-1),k=1) + np.diag(np.ones(N-1),k=-1))
    V_diag = np.diag(V)
    L_diag = np.diag(np.ones(N) * l*(l+1)/(x**2))
    
    H = tridiag + V_diag + L_diag
    
    eigvals, eigvecs = np.linalg.eig(H)
    
    # sort eigvals and eigvecs
    sorted_i = np.argsort(eigvals)
    eigvals = eigvals[sorted_i]
    eigvecs = eigvecs[:, sorted_i]
    eigvecs = eigvecs.T
    
    return eigvals, eigvecs


# e1 = SE_solver(V_WS(x), 0)
# print((hc**2)*e1[0][0] / (2*mc2))


## l = 0,1,2 -> s,p,d
# u_00 = SE_solver(V_WS(x), 0)[1][0]  # [i][j]:[eigvals/or/eigvecs][index]


for n in range(2):
    for l in range(2):
        print(f"\E_{n}{l} = ", (hc**2)*SE_solver(V_WS(x),l)[0][n] / (2*mc2), "MeV")
        plt.plot(x, 5*SE_solver(V_WS(x),l)[1][n] + SE_solver(V_WS(x),l)[0][n], label=f"u_{n}{l}(r)")
plt.plot(x,V_WS(x), color="black", label="V_WS(r)")
plt.xlabel("r [fm]")
plt.ylabel("E [MeV]")
plt.title("Reduced radial wavefunctions in Woods Saxon potential")
plt.legend()
plt.show()