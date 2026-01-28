import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg




hbar = 1


class OneParticleArray():
	
	@staticmethod
	def diagonalize(matrix):
		eigvals, eigvecs = np.linalg.eigh(matrix)
		return eigvals, eigvecs
	
	
	def __init__(self, L = 6, V = -1, eps1 = 20):
		H0 = np.diag(np.full(L-1, V), k = 1) + np.diag(np.full(L-1, V), k = -1)
		H = H0.copy()
		H[0][0] = eps1
		
		self.H0 = H0
		self.H = H
		self.L = L
		

	def psi(self, t):
		"""
		E_l: eigevelues to H matrix
		H_vecs: eigenvectors to H matrix
		H0_vecs: eigvectors to H0 matrix (hamiltonian t=0)
		
		c_n^l = H_vecs[:,l][n]
		c_m^l = H_vecs[:,l][m]
		c_m(0) = H0_vecs[:,0][m]
		
		E_l = E_l[i]
		"""

		E_l = self.diagonalize(self.H)[0]
		
		H_vecs = self.diagonalize(self.H)[1]
		H0_vecs = self.diagonalize(self.H0)[1]

		c_ns = np.full(self.L, np.nan, dtype=complex)	# [c_1(t), c_2(t), ...]
		
		for n in range(len(c_ns)):
			Sl = 0
			for l in range(len(E_l)):
				Sm = 0
				for m in range(self.L):
					Sm += H_vecs[:,l][m] * H0_vecs[:,0][m]
				Sl += np.exp(-1j * E_l[l] * t / hbar) * H_vecs[:,l][n] * Sm
			c_ns[n] = Sl
			
		return c_ns
	

	def rho(self,t):
		return [self.psi(t)[i] * np.conjugate(self.psi(t)[i]) for i in range(len(self.psi(t)))]
	
	
	def plot(self, t):
		tvals = t
		func = [self.rho(t) for t in tvals]
		plot = plt.plot(t, func)
		return plot


t_vals = np.linspace(0, 100, 1000)


test = OneParticleArray()
test.plot(t_vals)
plt.show()




class TwoParticle:
	@staticmethod
	def delta(a,b):
		return int(a==b)
	
	@staticmethod
	def diagonalize(matrix):
		eigvals, eigvecs = np.linalg.eigh(matrix)
		return eigvals, eigvecs

	def __init__(self, V, eps1, L, D):
		self.V = V
		self.eps1 = eps1
		self.L = L
		self.D = D
		self.u, self.v = np.linalg.eigh(self.Hamil())
		self.u0, self.v0 = np.linalg.eigh(self.Hamil(c0=True))
		

	def Hamil(self, c0 = bool):
		"""
		c0 = True -> H(t=0)
		c0 = False -> H(t>t)
		"""
		L = self.L
		V = self.V
		
		U_ngeq1 = np.zeros(L**2)
			
		for i in range(1, L**2):
			U_ngeq1[i] = 15
		#print(U_ngeq1)
		
		if c0 == True:
			H1 = np.diag(np.full(L-1, V), k = 1) + np.diag(np.full(L-1, V), k = -1)
			H1 = H1.copy()
			#print(2*self.eps1)
			H1[0][0] =  self.eps1

			H0 = np.zeros((L**2,L**2))
		
			for n in range(L):
				for m in range(L):
					for nprime in range(L):
						for mprime in range(L):
							val = H1[n][m] * self.delta(nprime,mprime) + self.delta(n,m) * H1[nprime][mprime] + U_ngeq1[n] * self.delta(n,nprime) * self.delta(m,mprime) * self.delta(n,m)
							N = L*nprime + n
							M = L*mprime + m
							H0[N][M] = val
			return H0
		else:
			H1 = np.diag(np.full(L-1, V), k = 1) + np.diag(np.full(L-1, V), k = -1)
			H1 = H1.copy()
			eps1 = self.eps1 + self.D
			H1[0][0] = eps1
			

			H = np.zeros((L**2,L**2))
		
			for n in range(L):
				for m in range(L):
					for nprime in range(L):
						for mprime in range(L):
							val = H1[n][m] * self.delta(nprime,mprime) + self.delta(n,m) * H1[nprime][mprime] + U_ngeq1[n] * self.delta(n,nprime) * self.delta(m,mprime) * self.delta(n,m)
							N = L*nprime + n
							M = L*mprime + m
							H[N][M] = val
				
			return H
		

	def psi(self):
		"""
		E_l: eigevelues to H matrix
		H_vecs: eigenvectors to H matrix
		H0_vecs: eigvectors to H0 matrix (hamiltonian t=0)
		
		c_n^l = H_vecs[:,l][n]
		c_m^l = H_vecs[:,l][m]
		c_m(0) = H0_vecs[:,0][m]
		
		E_l = E_l[i]
		"""

		#c_nnprime = np.full(self.L**2, np.nan, dtype=complex)	# [c_11(t), c_12(t), ...] = {c_nn'(t)}
		t = np.linspace(0,20, 1000)
		
		for n in range(self.L):
			rho = 0
			for nprime in range(self.L):
				Sl = 0
				for lam, E in enumerate(self.u):
					v = self.v[:,lam]	# entire c_nn'
					v_slice = v[n*self.L:n*self.L + self.L]
					v0_2 = self.v0[:,0]
					Sl += np.exp(-1j * E * t / hbar) * v_slice[nprime] * sum(v[m]*v0_2[m] for m in range(self.L**2))
				rho += abs(Sl)**2
			plt.plot(t, rho)
		plt.show()
	
	

	



obj = TwoParticle(V=-1,eps1=-15,L=6,D=15)
H0 = obj.Hamil(True)

# plt.imshow(H0)
# plt.colorbar()
# plt.show()

# Ht = obj.Hamil(False)
# plt.imshow(Ht)
# plt.colorbar()
# plt.show()


obj.psi()







 