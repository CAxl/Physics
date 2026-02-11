import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

class SPHsystem:
    def __init__(self, N, dim, kernel, gamma = 1.4):
        self.N = N # number of particles
        self.dim = dim # dimension of the problem
        self.kernel = kernel
        self.gamma = gamma

        self.S = np.zeros((N, 2*dim + 2)) # s_i (row) = [r_i (dim) | v_i (dim) | rho_i (1) | e_i (1)]
        self.m = np.ones(N) # all masses equal

    # getters
    # defines index convention as:
    # (dim==3): S[i][0] = x_i, S[i][1] = y_i, ..., S[i][-2] = \rho_i, S[i][-1] = e_i
    # S[i] = r[i] = [x,y,z]
    @property
    def r(self):
        return self.S[:,:self.dim]
    
    @property
    def v(self):
        return self.S[:,self.dim:2*self.dim]
    
    @property
    def rho(self):
        return self.S[:,2*self.dim]
    
    @property
    def e(self):
        return self.S[:,2*self.dim + 1]
    

    def geom(self):
        """
        black magic

        (in test case this looks correct)
        """
        ri = self.r[:,np.newaxis,:] # ri[i,0,:] = ri
        rj = self.r[np.newaxis,:,:] # rj[0,j,:] = rj

        rij = ri - rj  # drij[i,j,:] = ri - rj   (relative vector)
        rij_norm = np.linalg.norm(rij, axis=2) # (relative distance)

        return rij, rij_norm
    
    # density summation
    def density_summation(self):
        rij, rij_norm = self.geom()
        W = self.kernel.W(rij_norm)  # W(|r_i - r_j|)

        # fill state vector with densities (rho == S[:,2*self.dim])
        self.S[:,2*self.dim] = np.sum(self.m * W, axis = 1)

    # Equations of state
    def pressure(self):
        return (self.gamma - 1) * self.rho * self.e
    
    def sound_speed(self):
        return np.sqrt((self.gamma - 1) * self.e)


class cubicSplineKernel:
    """
    Implements the piecewise cubic spline kernel and its analytical derivative
    (Gaussian looking function, but goes to zero explicitly)
    
    measures the "region of influence" particle at x_i exerts on particle at x_j
    """

    def __init__(self, dim, h):
        self.h = h

        if dim == 1:
            self.a_d = 1/h 
        elif dim == 2:
            self.a_d = 15 / (7*np.pi*h**2)
        elif dim == 3:
            self.a_d = 3 / (2*np.pi*h**3)
    
    def W(self, rij_norm):
        R = rij_norm/self.h  # R = |x_i-x_j|/h
        f1 = lambda R: (2/3) - R**2 + 0.5*R**3
        f2 = lambda R: (1/6) * (2 - R)**3

        return self.a_d * np.piecewise(R, [(R>= 0) & (R<1), (R>=1) & (R<=2)], [f1,f2,0.0])


    def gradW(self, rij, rij_norm):
        R = rij_norm/self.h

        f1 = lambda R: (-2 + 1.5*R) * (rij / self.h**2)
        f2 = lambda R: -0.5*(2 - R)**2 * rij / (rij_norm * self.h)

        gradW = self.a_d * np.piecewise(R, [(R>= 0) & (R<1), (R>=1) & (R<=2)], [f1,f2,0.0])
        
        # protect against nan where |r_i - r_j| = 0 (diagonal)
        np.fill_diagonal(gradW, 0.0)

        return gradW




dim = 3
kernel = cubicSplineKernel(dim, h=1.0)
sys = SPHsystem(3,dim,kernel)



x = np.array([0,2,4])
y = np.array([0,3,1])
z = np.array([0,0,0])

r = np.column_stack((x,y,z))    # -> one particle at (x,y) = (0,0), one at (2,3) and one at (4,1)
sys.S[:,:dim] = r

print(sys.S)

x = np.linspace(-2, 5,100)
y = np.linspace(-2, 5,100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
Wtot = np.zeros_like(X)

for r_i in r:
    dx = X - r_i[0]
    dy = Y - r_i[1]
    dz = Z - r_i[2]
    R  = np.sqrt(dx**2 + dy**2 + dz**2)
    Wtot += kernel.W(R)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Wtot, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()



