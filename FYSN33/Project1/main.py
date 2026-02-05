import numpy as np
import matplotlib.pyplot as plt


class SPHsystem:
    def __init__(self, N, kernel, e0 = 2.5):
        self.N = N # number of particles
        self.kernel = kernel
        
        self.S = np.zeros((N,4)) # s_i = [x_i, v_i, \rho_i, e_i]
        self.m = np.ones(N) # all masses equal
        self.S[:,3] = e0

    # getters
    # defines index convention as:
    # S[i][0] = x_i, S[i][1] = v_i, S[i][2] = \rho_i, S[i][3] = e_i
    # in other words, initialize S using this convention please!
    @property
    def x(self):
        return self.S[:, 0]

    @property
    def v(self):
        return self.S[:, 1]

    @property
    def rho(self):
        return self.S[:, 2]

    @property
    def e(self):
        return self.S[:, 3]
    
    # define the geometry, dx and r = |x_i - x_j|. How to move to 3D in r?
    # note that dx is signed!
    def geom(self):
        # broadcasting (same logic as double loop i,j\in x_linspace)
        dx = self.x[:,None] - self.x[None,:] 
        r = np.abs(dx)
        print(np.shape(r))

        return dx, r
    
    # density summation
    def density_summation(self):
        dx, r = self.geom()
        W = self.kernel.W(r)

        # fill state vector with densities (rho[:] == S[:,2])
        self.rho[:] = W @ self.m

        # check for numerical stability
        assert np.all(self.rho > 0) and np.isfinite(self.rho).all()
        return self


class cubicSplineKernel:
    # implements the cubic spline kernel and its analytical derivative
    # (Gaussian looking function, but goes to zero explicitly)
    # It measures the "region of influence" particle at x_i exerts on particle at x_j

    def __init__(self, h):
        self.h = h
        self.a_d = 1/h # for now
    
    def W(self,r):
        R = np.abs(r)/self.h    # R_ij = |x_i-x_j|/h
        f1 = lambda R: (2/3) - R**2 + 0.5*R**3
        f2 = lambda R: (1/6) * (2 - R)**3

        return self.a_d * np.piecewise(R, [(R>= 0) & (R<1), (R>=1) & (R<=2)], [f1,f2,0.0])

    def gradW(self, dx, r):

        R = np.abs(r)/self.h
        f1 = -2*R + 1.5 * R**2
        f2 = -0.5 * (2 - R)**2

        dWdr = self.a_d * np.piecewise(R, [(R>= 0) & (R<1), (R>=1) & (R<=2)], [f1,f2,0.0])

        # check diags not nan
        gradW = dWdr * np.sign(dx)
        assert np.allclose(np.diag(gradW), 0.0)

        return gradW
    

# NS operator
class NavierStokes1D:
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    def pressure(self,system):
        rho = system.S[:,2] # extract each density
        e = system.S[:,3] # extract each internal energy
        
        press = (self.gamma - 1) * rho * e
        
        # asserts here

        return press







