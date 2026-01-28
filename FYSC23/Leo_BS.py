import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as lin


class TimeEvo:

    def __init__(self, n, V, eps, t):
        self.n = n
        self.V = V
        self.eps = eps
        self.t = np.linspace(0, t, 1000)
        
    
    def matrix_gen(self, c0 = bool, c = bool):
        M = np.zeros((self.n, self.n))
        for element in range(0, self.n-1):
            M[element][element+1] = M[element+1][element] = self.V
        if c0 == True:
            return M
        if c == True:
            M[0][0] = self.eps
            return M


    def eigvals_vecs(self):
        hbar = 1
        c0 = self.matrix_gen(c0=True)
        c = self.matrix_gen(c=True)
        c0vals, c0vecs = lin.eigh(c0)
        cvals, cvecs = lin.eigh(c)
        for n in range(self.n):
            ct = 0
            for lam, E in enumerate(cvals):
                v = cvecs[:,lam]
                v0 = c0vecs[:,0]
                ct += np.exp(-1j*E*self.t)*v[n]*sum(v[m]*v0[m] for m in range(self.n))
            plt.plot(self.t, abs(ct)**2, label=f'$\\rho_{n+1}(t)$')
        plt.title(f'$\\epsilon_1 = {self.eps}$')
        plt.xlabel('$t$')
        plt.ylabel('$\\rho_n(t)$')
        plt.legend()
        plt.show()
        
    def plotting(self):
        hbar = 1
        c0 = self.matrix_gen(c0=True)
        c = self.matrix_gen(c=True)
        c0vals, c0vecs = lin.eigh(c0)
        cvals, cvecs = lin.eigh(c)
        lst = []
        for n in range(self.n):
            ct = 0
            for lam, E in enumerate(cvals):
                v = cvecs[:,lam]
                v0 = c0vecs[:,0]
                ct += np.exp(-1j*E*self.t)*v[n]*sum(v[m]*v0[m] for m in range(self.n))
            lst.append(abs(ct)**2)
        return lst



t = np.linspace(0,20,1000)

fig, axs = plt.subplots(2)
lst = [-2,2]
for ind, val in enumerate(lst):
    C_ = TimeEvo(6, -1, val, 20)
    R = C_.plotting()
    for i in range(6):
        axs[ind].plot(t, R[i], label=f'$\\rho_{i+1}$')

axs[0].set_ylabel('$\\rho_n(t)$')
axs[1].set_ylabel('$\\rho_n(t)$')
axs[1].set_xlabel('$t$')

axs[1].legend(bbox_to_anchor=(1,1))
plt.savefig('fig.png', bbox_inches='tight')


