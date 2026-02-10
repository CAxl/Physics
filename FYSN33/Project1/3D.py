import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter

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
    @property
    def r(self):
        return self.S[:,:self.dim]
    
    @property
    def v(self):
        return self.S[:,self.dim:2*self.dim]
    
    def rho(self):
        return self.S[:,2*self.dim]
    
    def e(self):
        return self.S[:,2*self.dim + 1]
    

    def geom(self):
        """
        black magic

        (in test case this looks correct)
        """
        ri = self.r[:,np.newaxis,:] # ri[i,0,:] = ri
        rj = self.r[np.newaxis,:,:] # rj[0,j,:] = rj

        drij = ri - rj  # drij[i,j,:] = ri - rj
        r = np.linalg.norm(drij, axis=2)

        return drij, r
    
    # density summation
    def density_summation(self):
        drij, r = self.geom()
        W = self.kernel.W(r)

        # fill state vector with densities (rho == S[:,2*self.dim])
        self.rho = np.sum(self.m * W, axis = 1)

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
            self.a_d = 1/h # 1D ONLY!!!
        elif dim == 2:
            self.a_d = 17 / (7*np.pi*h**2)
        elif dim == 3:
            self.a_d = 3 / (2*np.pi*h**3)
    
    def W(self,r):
        R = np.abs(r)/self.h  # R_ij = |x_i-x_j|/h
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
    

# # NS operator
# class NavierStokes1D:

#     def __init__(self, alpha = 1.0, beta = 1.0):
#         self.alpha = alpha  # for visc
#         self.beta = beta


#     def artificial_visc(self, system):
#         dx, r = system.geom()
        
#         vij = system.v[:,None] - system.v[None,:] # v_ij = v_i - v_j
#         xij = dx                             # x_ij = x_i - x_j (already calculated in geom())
#         varphi = 0.1 * system.kernel.h

#         phi_ij = system.kernel.h * vij * xij / (r**2 + varphi**2) # r = |x_ij| (geom())

#         c = system.sound_speed()
#         cij_bar = 0.5 * (c[:,None] + c[None,:])

#         rhoij_bar = 0.5 * (system.rho[:,None] + system.rho[None,:])

#         Pi_ij = (-self.alpha * cij_bar * phi_ij + self.beta * phi_ij**2) / rhoij_bar

#         # mask the viscosity according condition vij * xij >=0 (theory)
#         vij_dot_xij = xij*vij
#         Pi_ij[vij_dot_xij >= 0] = 0.0

#         return Pi_ij

#     def momentum_equation(self, system):
#         dx, r = system.geom()

#         # collect kernel gradient and pressure vector
#         gradW = system.kernel.gradW(dx, r)
#         P = system.pressure()

#         # viscosity
#         Pi_ij = self.artificial_visc(system)

#         # term in parenthesis in momentum equation
#         parenthesis = (P[:,None] / system.rho[:,None]**2 # P_i/rho_i²
#                       +P[None,:] / system.rho[None,:]**2 # P_j/rho_j²
#                       +Pi_ij)
        
#         dvdt = -np.sum(system.m[None,:] # m_j
#                      * parenthesis * gradW, axis=1) # axis = 1 -> (N,)
        
#         return dvdt

#     def energy_equation(self, system):
#         dx, r = system.geom()

#         gradW = system.kernel.gradW(dx, r)
#         P = system.pressure()

#         # viscosity 
#         Pi_ij = self.artificial_visc(system)

#         vij = system.v[:,None] - system.v[None,:]
        
#         parenthesis = (P[:,None] / system.rho[:,None]**2 # P_i/rho_i²
#                      + P[None,:] / system.rho[None,:]**2 # P_j/rho_j²
#                      + Pi_ij)


#         dedt = (1/2) * np.sum(system.m[None,:] # m_j
#                             * parenthesis * vij * gradW, axis = 1) 
        
#         return dedt
    

# # (put into Solver class eventually)
# def RHS(t, S_flat, system, NSequations):
#     """
#     RHS of the ODE:

#     f(y,t) = dy/dt

#     dy/dt is dS/dt = [\dot{x}, \dot{v}, \dot{rho}, \dot{e}]

#     Computes dS/dt given the current flattedned state vector
#     RHS required by scipy.solve_ivp 
#     """

#     # rebuild the State vector (matrix) as the (N,4) shape
#     S = S_flat.reshape(system.N, 4) 
#     system.S[:] = S # update object

#     # update density (summation, not continuity)
#     system.density_summation()

#     # compute time derivatives
#     dxdt = system.v
#     dvdt = NSequations.momentum_equation(system)
#     dedt = NSequations.energy_equation(system)

#     dSdt = np.column_stack((dxdt,
#                             dvdt,
#                             np.zeros(system.N), # index [2] == rho[:] not updated, recomputed
#                             dedt)) # obs order has to match S index convention
    
#     return dSdt.flatten()


