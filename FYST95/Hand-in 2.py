import numpy as np
import matplotlib.pyplot as plt


N = 3
eps = 2

# 2\Omega = degeneract -> \Omega = (2j+1)/2
Omega1 = 3	# f_5/2
Omega2 = 5	# g_9/2

def H_matrix(G):
	dim = N + 1
	diag = np.zeros(dim)
	low  = np.zeros(dim-1)  # sub-diagonal (k = -1)

	# diagonal elements
	for n1 in range(dim):
		n2 = N - n1
		diag[n1] = (N - 2*n1)*eps - G*( n1*(Omega1 + 1 - n1) + n2*(Omega2 + 1 - n2))

	# lower diagonal: H[n1, n1-1] = <n1|H|n1-1> = <n1-1|H|n1> (Hermitian)
	for n1 in range(1, dim):
		n2 = N - n1
		low[n1-1] = -G*np.sqrt(n1 * (Omega1 + 1 - n1) *(n2 + 1) * (Omega2 - n2))

	# build full matrix
	H = np.diag(diag) + np.diag(low, k=-1) + np.diag(low, k=+1)
	
	return H

H = H_matrix(1)

# plt.matshow(H)
# plt.colorbar()
# plt.show()


Gscan = np.linspace(0,1, 100)
E0 = []
E1 = []
E2 = []
E3 = []

dim = N+1
coeffs0 = np.zeros((dim, len(Gscan)))
coeffs1 = np.zeros((dim, len(Gscan)))
coeffs2 = np.zeros((dim, len(Gscan)))
coeffs3 = np.zeros((dim, len(Gscan)))

E_exc = []

for iG, G in enumerate(Gscan):
	H = H_matrix(G)
	
	eigvals, eigvecs = np.linalg.eigh(H)
	
	E0.append(eigvals[0])
	E1.append(eigvals[1])
	E2.append(eigvals[2])
	E3.append(eigvals[3])
	
	E_exc.append(eigvals[1]-eigvals[0])
	
	# store coefficients of 0th, 1st, ... eigvecs
	coeffs0[:,iG] = eigvecs[:,0] # first column
	coeffs1[:,iG] = eigvecs[:,1]
	coeffs2[:,iG] = eigvecs[:,2]
	coeffs3[:,iG] = eigvecs[:,3]
	
	
E0 = np.array(E0)
E1 = np.array(E1)
E2 = np.array(E2)
E3 = np.array(E3)

E_exc = np.array(E_exc)

psi0 = np.array(coeffs0)
print(psi0[0,:]) # zeroth c_{n_1=0}

psi1 = np.array(coeffs1)

for n1 in range(dim):
	plt.plot(Gscan, np.abs(psi0[n1,:])**2, label=f"n1={n1}")
plt.xlabel("Interaction strength, G [MeV]")
plt.ylabel("Squared amplitude |c_{n1}(G)|^2")
plt.grid()
plt.legend()
plt.show()

# # plot k = 1 wavefunctions (|c_{n_1}^{(k=1)}(G)|^2)
# for n1 in range(dim):
# 	plt.plot(Gscan, np.abs(psi1[n1,:])**2, label=f"n1={n1}")
# plt.show()


plt.plot(Gscan, E0, label="E0")
plt.plot(Gscan, E1, label="E1")
plt.plot(Gscan, E2, label="E2")
plt.plot(Gscan, E3, label="E3")


plt.xlabel("Interaction strength, G [MeV]")
plt.ylabel("Energy, E [MeV]")
plt.grid()
plt.legend()
plt.show()


plt.plot(Gscan, E_exc, label="E_exc")
plt.axhline(3.4, color='black', linestyle='--', label="Exp. 0$^+$ at 3.4 MeV")
plt.xlabel("G [MeV]")
plt.ylabel("E_exc(G) = E_1(G)-E_0(G) [MeV]")
plt.legend()
plt.grid()
plt.show()




