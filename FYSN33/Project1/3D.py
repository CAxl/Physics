import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter

class SPHsystem:
    def __init__(self, N, kernel, gamma = 1.4):
        self.N = N # number of particles
        self.kernel = kernel
        self.gamma = gamma
        
        self.S = np.zeros((N,4)) # s_i = [x_i, v_i, \rho_i, e_i]
        self.m = np.ones(N) # all masses equal        

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
        """
        Docstring for geom
        
        calculates the (signed) separation, dx = x_i-x_j, between particles and
        the (unsigned) distance r_ij = |x_i-x_j| between particles

        given that S[:,0] has been initialized as a position vector
        """
        
        dx = self.x[:,None] - self.x[None,:] 
        r = np.abs(dx)

        return dx, r
    
    # density summation
    def density_summation(self):
        dx, r = self.geom()
        W = self.kernel.W(r)

        # fill state vector with densities (rho[:] == S[:,2])
        self.S[:,2] = np.sum(self.m * W, axis = 1)

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

    def __init__(self, h):
        self.h = h
        self.a_d = 1/h # 1D ONLY!!!
    
    def W(self,r):
        R = np.abs(r)/self.h    # R_ij = |x_i-x_j|/h
        f1 = lambda R: (2/3) - R**2 + 0.5*R**3
        f2 = lambda R: (1/6) * (2 - R)**3

        return self.a_d * np.piecewise(R, [(R>= 0) & (R<1), (R>=1) & (R<=2)], [f1,f2,0.0])

    def gradW(self, dx, r):

        R = np.abs(r)/self.h
        f1 = lambda R: -2*R + 1.5 * R**2
        f2 = lambda R: -0.5 * (2 - R)**2

        dWdr = (self.a_d / self.h) * np.piecewise(R, [(R>= 0) & (R<1), (R>=1) & (R<=2)], [f1,f2,0.0])

        # check diags not nan
        gradW = dWdr * np.sign(dx)

        return gradW
    

# NS operator
class NavierStokes1D:

    def __init__(self, alpha = 1.0, beta = 1.0):
        self.alpha = alpha  # for visc
        self.beta = beta


    def artificial_visc(self, system):
        dx, r = system.geom()
        
        vij = system.v[:,None] - system.v[None,:] # v_ij = v_i - v_j
        xij = dx                             # x_ij = x_i - x_j (already calculated in geom())
        varphi = 0.1 * system.kernel.h

        phi_ij = system.kernel.h * vij * xij / (r**2 + varphi**2) # r = |x_ij| (geom())

        c = system.sound_speed()
        cij_bar = 0.5 * (c[:,None] + c[None,:])

        rhoij_bar = 0.5 * (system.rho[:,None] + system.rho[None,:])

        Pi_ij = (-self.alpha * cij_bar * phi_ij + self.beta * phi_ij**2) / rhoij_bar

        # mask the viscosity according condition vij * xij >=0 (theory)
        vij_xij = xij*vij
        Pi_ij[vij_xij >= 0] = 0.0

        return Pi_ij

    def momentum_equation(self, system):
        dx, r = system.geom()

        # collect kernel gradient and pressure vector
        gradW = system.kernel.gradW(dx, r)
        P = system.pressure()

        # viscosity
        Pi_ij = self.artificial_visc(system)

        # term in parenthesis in momentum equation
        parenthesis = (P[:,None] / system.rho[:,None]**2 # P_i/rho_i²
                      +P[None,:] / system.rho[None,:]**2 # P_j/rho_j²
                      +Pi_ij)
        
        dvdt = -np.sum(system.m[None,:] # m_j
                     * parenthesis * gradW, axis=1) # axis = 1 -> (N,)
        
        return dvdt

    def energy_equation(self, system):
        dx, r = system.geom()

        gradW = system.kernel.gradW(dx, r)
        P = system.pressure()

        # viscosity 
        Pi_ij = self.artificial_visc(system)

        vij = system.v[:,None] - system.v[None,:]
        
        parenthesis = (P[:,None] / system.rho[:,None]**2 # P_i/rho_i²
                     + P[None,:] / system.rho[None,:]**2 # P_j/rho_j²
                     + Pi_ij)


        dedt = (1/2) * np.sum(system.m[None,:] # m_j
                            * parenthesis * vij * gradW, axis = 1) 
        
        return dedt
    


# for Sod Shock and for now free function:
# (put into Solver class eventually)
def RHS(t, S_flat, system, NSequations):
    """
    RHS of the ODE:

    f(y,t) = dy/dt

    dy/dt is dS/dt = [\dot{x}, \dot{v}, \dot{rho}, \dot{e}]

    Computes dS/dt given the current flattedned state vector
    RHS required by scipy.solve_ivp 
    """

    # rebuild the State vector (matrix) as the (N,4) shape
    S = S_flat.reshape(system.N, 4) 
    system.S[:] = S # update object

    # update density (summation, not continuity)
    system.density_summation()

    # compute time derivatives
    dxdt = system.v
    dvdt = NSequations.momentum_equation(system)
    dedt = NSequations.energy_equation(system)

    dSdt = np.column_stack((dxdt,
                            dvdt,
                            np.zeros(system.N), # index [2] == rho[:] not updated, recomputed
                            dedt)) # obs order has to match S index convention
    
    return dSdt.flatten()


