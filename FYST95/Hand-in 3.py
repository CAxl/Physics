import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import scipy



# ------------------------------------- 3.2a --------------------------------------------------

# hbaromega0 = 10

# def E100(delta):
# 	return hbaromega0 * ((5/2) + (1/3)*delta)

# def E001(delta):
# 	return hbaromega0 * ((5/2) - (2/3)*delta)

# delta = np.linspace(-1, 1, 1000)


# # vertical line at δ = 0
# plt.axvline(0, linestyle=":", color = "gray", alpha = 0.7, linewidth=2)

# # horizontal line at the average (centroid)
# E_avg = hbaromega0 * (5/2)
# plt.axhline(E_avg, linestyle="--", color = "gray", alpha = 0.7, linewidth=2)


# plt.plot(delta, E100(delta), label="E_100 = E_010")
# plt.plot(delta, E001(delta), label="E_001")
# plt.xlabel("\delta")
# plt.ylabel("Energy")
# plt.legend()
# plt.grid()
# plt.show()




# # try to generalize for fun

# def E(nx, ny, nz, delta):
#     perp = hbaromega0 * ((5/2) + (1/3)*delta)
#     z = hbaromega0 * ((5/2) - (2/3)*delta)
	

#     E = perp * (nx + ny + 1) + z * (nz + 1/2)
	
#     return E


# # N = 1
# E100 = E(1,0,0, delta)
# E010 = E(0,1,0, delta)
# E001 = E(0,0,1, delta)

# plt.plot(delta, E100)
# plt.plot(delta, E010)
# plt.plot(delta, E001)

# # N = 2 (d = 6)
# E110 = E(1,1,0, delta)
# E101 = E(1,0,1, delta)
# E011 = E(0,1,1, delta)
# E200 = E(2,0,0, delta)
# E020 = E(0,2,0, delta)
# E002 = E(0,2,0, delta)

# plt.plot(delta, E110)
# plt.plot(delta, E101)
# plt.plot(delta, E011)
# plt.plot(delta, E200)
# plt.plot(delta, E020)
# plt.plot(delta, E002)

# N = 3 (d = 10)
# plt.plot(delta, E(1,1,1, delta))
# plt.plot(delta, E(2,1,0, delta))
# plt.plot(delta, E(2,0,1, delta))
# plt.plot(delta, E(0,2,1, delta))
# plt.plot(delta, E(0,1,2, delta))
# plt.plot(delta, E(1,2,0, delta))
# plt.plot(delta, E(1,0,2, delta))
# plt.plot(delta, E(3,0,0, delta))
# plt.plot(delta, E(0,3,0, delta))
# plt.plot(delta, E(0,0,3, delta))


# plt.show()


# ------------------------------------ 3.2b -------------------------------------
# def E10(delta):
# 	return hbaromega0 * ((5/2) - (2/3)*delta)

# def E01(delta):
# 	return hbaromega0 * ((5/2) + (1/3)*delta)


# mu = 5

# def block1(delta):
# 	arr = np.array([[E10(delta), mu/np.sqrt(2)],
# 					[mu/np.sqrt(2), E01(delta) + mu/2]])
	
# 	return arr


# def block2(delta):
# 	arr = np.array([[E10(delta), -mu/np.sqrt(2)],
# 					[-mu/np.sqrt(2), E01(delta) + mu/2]])
	
# 	return arr

# # deformation range
# deltas = np.linspace(-1, 1, 1000)

# # storage
# E_p32 = []   # \Omega = \pm3/2
# E_m32 = []
# E_p12_1 = [] # \Omega = +1/2
# E_p12_2 = []
# E_m12_1 = [] # \Omega = -1/2
# E_m12_2 = []

# for d in deltas:
#     # Omega = \pm 3/2
#     E_p32.append((E01(d) - mu/2) / hbaromega0) 
#     E_m32.append((E01(d) - mu/2) / hbaromega0) 

#     # Omega \pm 1/2 blocks
#     e1 = np.linalg.eigvalsh(block1(d))
#     e2 = np.linalg.eigvalsh(block2(d))

#     E_p12_1.append(e1[0] / hbaromega0)
#     E_p12_2.append(e1[1] / hbaromega0)
#     E_m12_1.append(e2[0] / hbaromega0)
#     E_m12_2.append(e2[1] / hbaromega0)

# plt.figure(figsize=(7,5))

# plt.plot(deltas, E_p32, 'k--', label=r'$\Omega=\pm 3/2$')
# plt.plot(deltas, E_p12_1, 'b')
# plt.plot(deltas, E_p12_2, 'b', label=r'$\Omega=+1/2$')
# plt.plot(deltas, E_m12_1, 'r--')
# plt.plot(deltas, E_m12_2, 'r--', label=r'$\Omega=-1/2$')

# plt.xlabel(r'$\delta$')
# plt.ylabel(r'$E/\hbar\omega_0$')
# plt.legend()
# plt.tight_layout()
# plt.show()




# -------------------------------------- 3.3 GCM ------------------------------------------------


mu = 0.1
hbar = 1.0
m = 1.0

# generator coordinates x_i
# sensitive to dense coordinates (-1,1, 100) didn't even almost work
xis = np.linspace(-3, 3, 31)
N = len(xis)
print("N = ", N)

# build norm matrix
norm_mat = np.zeros((N, N))

for i in range(N):
	for j in range(N):
		norm_mat[i, j] = np.exp(-(xis[i] - xis[j])**2 / (2 * mu))


# build hamiltonian matrix
H_mat = np.zeros((N,N))

k2 = 1
k4 = 0

for i in range(N):
	for j in range(N):
		xi = xis[i]
		xj = xis[j]
		dx = xi - xj
		xp = xi + xj
		
		overlap = np.exp(-dx**2/(2*mu))
		
		t1 = (-hbar**2/(2*m)) * (1/mu**2) * (dx**2 - mu)
		t2 = (k2/8) * (xp**2 + mu)
		t3 = (k4/16) * ((xp**4) + 6*mu*(xp**2) + 3*mu**2)
		
		H_mat[i,j] = (t1 + t2 + t3) * overlap


# generalized eigval problem in scipy
E, C = scipy.linalg.eigh(H_mat, norm_mat)

xgrid = np.linspace(-3, 3, 1000)
plt.figure()

def psi(x, xi):
	return (2 / (np.pi * mu))**0.25 * np.exp(-(x - xi)**2 / mu)

def V1(x):
	return (1/2)*k2*x**2

def V2(x):
	return (1/2)*k2*x**2 + k4*x**4


plt.plot(xgrid, V1(xgrid),color = "black", label="V(x)")

for n in range(4):
	Psi = np.zeros_like(xgrid)
	for i in range(N):
		Psi += C[i, n] * psi(xgrid, xis[i]) # \ket{\Psi} = \sum_i c_i\ket{\psi_i} (c@psi)

	# normalize for plotting
	Psi /= np.sqrt(np.trapz(Psi**2, xgrid))

	plt.plot(xgrid, np.abs(Psi)**2 + E[n], label=f"n={n}, E={E[n]:.3f}")
	

plt.title(f"k2 = {k2}, k4 = {k4}")
plt.xlabel("x")
plt.ylabel(r"$|\Psi_n(x)|^{2}$")
plt.legend()
plt.show()


print(E)

# O_inv = np.linalg.inv(norm_mat)
# HW_mat = O_inv @ H_mat

# E, C = np.linalg.eig(HW_mat)
# E = np.real(E)



# error analysis (can only compare using the analytical HO)

E_exact = np.array([0.5, 1.5, 2.5, 3.5])

Ns = [11, 21, 31, 41]	# basis size
errors = []



for N in Ns:
	# build norm and H matrices (same code as before)
	Nmat = np.zeros((N, N))
	Hmat = np.zeros((N, N))

	xis = np.linspace(-3, 3, N)

	for i in range(N):
		for j in range(N):
			dx = xis[i] - xis[j]
			xp = xis[i] + xis[j]
			overlap = np.exp(-dx**2/(2*mu))

			Nmat[i,j] = overlap

			t1 = (-hbar**2/(2*m))*(1/mu**2)*(dx**2 - mu)
			t2 = (k2/8)*(xp**2 + mu)

			Hmat[i,j] = (t1 + t2)*overlap

	E, _ = scipy.linalg.eigh(Hmat, Nmat)

	# take first 4 eigenvalues
	err = np.abs(E[:4] - E_exact)
	errors.append(err)

errors = np.array(errors)

for n in range(4):
	plt.plot(Ns, errors[:, n], marker='o', label=f'n={n}')

plt.yscale("log")
plt.xlabel("Basis size N")
plt.ylabel("Absolute energy error")
plt.legend()
plt.grid(True)
plt.show()




mus = np.linspace(0.05, 0.25, 10)
errors_mu = []

M = 31
xis = np.linspace(-3, 3, M)

for mu in mus:
    Nmat = np.zeros((M, M))
    Hmat = np.zeros((M, M))

    for i in range(M):
        for j in range(M):
            dx = xis[i] - xis[j]
            xp = xis[i] + xis[j]
            overlap = np.exp(-dx**2/(2*mu))

            Nmat[i,j] = overlap

            t1 = (-hbar**2/(2*m))*(1/mu**2)*(dx**2 - mu)
            t2 = (k2/8)*(xp**2 + mu)

            Hmat[i,j] = (t1 + t2)*overlap

    E, _ = scipy.linalg.eigh(Hmat, Nmat)
    errors_mu.append(np.abs(E[:4] - E_exact))

errors_mu = np.array(errors_mu)


for n in range(4):
    plt.plot(mus, errors_mu[:, n], marker='o', label=f'n={n}')

plt.xlabel(r'Gaussian width $\mu$')
plt.ylabel("Absolute energy error")
plt.legend()
plt.grid(True)
plt.show()



