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
        Think of every property as indexed [i,j,k]
        i = particle index treated
        j = neighbouring particle j \in {0,N}
        k = x,y,z

        (in test case this looks correct)
        """
        ri = self.r[:,None,:] # ri[i,0,:] = ri
        rj = self.r[None,:,:] # rj[0,j,:] = rj

        rij = ri - rj  # rij[i,j,:] = ri - rj   (relative vector)
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
        R = np.abs(rij_norm)/self.h  # R = |x_i-x_j|/h
        f1 = lambda R: (2/3) - R**2 + 0.5*R**3
        f2 = lambda R: (1/6) * (2 - R)**3

        return self.a_d * np.piecewise(R, [(R>= 0) & (R<1), (R>=1) & (R<=2)], [f1,f2,0.0])


    def gradW(self, rij, rij_norm):
        R = np.abs(rij_norm)/self.h

        gradW = np.zeros_like(rij)  # preallocate (N,N,dim)

        mask1 = (R >= 0) & (R < 1)
        mask2 = (R >= 1) & (R <=2)

        gradW[mask1] = self.a_d * (-2 + 1.5 * R[mask1][:,None]) * rij[mask1] / self.h**2
        gradW[mask2] = -self.a_d * 0.5 * (2 - R[mask2][:,None])**2 * rij[mask2] / (rij_norm[mask2][:,None] * self.h)      

        return gradW
    
class NSequations:
    def __init__(self, alpha = 1.0, beta = 1.0, selfgrav_flag = False):
        self.alpha = alpha
        self.beta = beta
        self.selfgrav_flag = selfgrav_flag

    def artifical_visc(self, system):
        rij, rij_norm = system.geom()

        # \vec{v}_{ij} = (v_{x,i} - v_{x,j}, v_{y,i} - v_{y,j}, v_{z,i} - v_{z,j})
        # vij[i,j,0] = x component
        # vij[i,j,1] = y component
        # vij[i,j,2] = z component
        # velocity differences (N,N,dim)
        vij = system.v[:,np.newaxis,:] - system.v[np.newaxis,:,:]
        vij_dot_rij = np.sum(vij*rij, axis=2)

        # phi_ij function
        varphi = 0.1 * system.kernel.h
        phi_ij = system.kernel.h * vij_dot_rij / (rij_norm**2 + varphi**2)

        # sound speed
        c = system.sound_speed()
        cij_bar = 0.5 * (c[:,None] + c[None,:])

        # density, \bar{rho}_ij
        rhoij_bar = 0.5 * (system.rho[:,None] + system.rho[None,:])
        
        # viscosity
        Pi_ij = (-self.alpha * cij_bar * phi_ij + self.beta * phi_ij**2) / rhoij_bar

        # mask the viscosity according condition vij * xij >=0 (theory)
        Pi_ij[vij_dot_rij >= 0] = 0.0

        return Pi_ij
    
    def selfgravity(self, system, G = 6.6743*1e-9):
        rij, rij_norm = system.geom()   # (N,N,dim), (N,N)
        h = system.kernel.h

        R = rij_norm / h

        # preallocate dphidr shape == (N,N)
        dphidr = np.zeros_like(rij_norm)

        # piecewise derivative dphi/dr
        mask1 = (R >= 0) & (R < 1)
        mask2 = (1 <= R) & (R < 2)
        mask3 = (R >= 2)

        # 0 <= R < 1
        dphidr[mask1] = (1/h**2) * ((4/3)*R[mask1] - (6/5)*R[mask1]**3 + (1/2)*R[mask1]**4)
        
        # 1 <= R < 2
        dphidr[mask2] = (1/h**2) * ((8/3)*R[mask2] - 3*R[mask2]**2 + (6/5)*R[mask2]**3 - (1/6)*R[mask2]**4 - 1/(15*R[mask2]**2))

        # R >= 2
        dphidr[mask3] = 1 / (rij_norm[mask3]**2) # computationally heavy part (eliminates zeros)

        # remove diagonal self-interaction terms
        np.fill_diagonal(dphidr, 0.0)

        
        # calculate gradient grad_iphi_ij:
        eps = 1e-12
        inv_r = 1.0 / (rij_norm + eps)  # avoid div zero
        gradphi_ij = dphidr[:,:,None] * rij * inv_r[:,:,None]

        # sum over j (axis1)
        dvdt_grav = - G * np.sum(system.m[None,:,None] * gradphi_ij, axis = 1)

        return dvdt_grav



    def momentum_equation(self,system):
        rij, rij_norm = system.geom()

        # collect kernel gradient and pressure from system
        gradW = system.kernel.gradW(rij,rij_norm)   # (N,N,dim)
        P = system.pressure()

        # collect viscosity from self
        Pi_ij = self.artifical_visc(system)

        # term in parenthesis in momentum equation
        parenthesis = (P[:,None] / system.rho[:,None]**2 # P_i/rho_i²
                      +P[None,:] / system.rho[None,:]**2 # P_j/rho_j²
                      +Pi_ij)   # (N,N)
        
        # gradW[i,j,k]
        # i = particle updated
        # j = neighboring particle
        # k = x,y,z component
        # \sum_j m_j(...)\nabla_iW_ij -> axis = 1 == j

        dvdt = -np.sum(system.m[None,:,None] * parenthesis[:,:,None] * gradW, axis=1) #[i,j,k] sum over j (axis 1)

        # add gravity if true
        if self.selfgrav_flag:
            dvdt = dvdt + self.selfgravity(system)

        return dvdt # (N,dim)


    def energy_equation(self, system):
        rij, rij_norm = system.geom()

        # collect gradW, pressure
        gradW = system.kernel.gradW(rij,rij_norm)
        P = system.pressure()

        # [i,j,k]
        # v_ik = v[:,None,:], v_jk = v[None,:,:]
        vij = system.v[:,None,:] - system.v[None,:,:] # (N,N,dim)
        
        # viscosity
        Pi_ij = self.artifical_visc(system)

        parenthesis = (P[:,None] / system.rho[:,None]**2 # P_i/rho_i²
                      +P[None,:] / system.rho[None,:]**2 # P_j/rho_j²
                      +Pi_ij)   # (N,N)
        
        # dot product (\vec{v}\cdot\nabla_iW_ij)
        vij_dot_gradW = np.sum(vij * gradW,axis=2) # sum over k (x,y,z) since dotproduct -> (N,N)

        dedt = 0.5 * np.sum(system.m[None,:] # m_j (no k:th index)
                          * parenthesis * vij_dot_gradW, axis = 1) # (N,)

        return dedt

def RHS(t, S_flat, system, NSequations):

    # rebuild state vector (matrix) as the (N, 2dim+2) shape
    S = S_flat.reshape(system.N, 2*system.dim + 2)
    system.S[:] = S # update object

    # update density
    system.density_summation()

    # compute time derivatives
    drdt = system.v # (N,dim)
    dvdt = NSequations.momentum_equation(system)
    dedt = NSequations.energy_equation(system)

    # preallocate dSdt with shape == S
    # dSdt = [\dot{r}(dim) | \dot{v} (dim) | \dot{rho} (1) | \dot{e} (1)]
    dSdt = np.zeros_like(system.S)

    # fill blocks
    dSdt[:, :system.dim] = drdt
    dSdt[:, system.dim:2*system.dim] = dvdt
    dSdt[:, -2] = 0.0 # rho not updated, recomputed
    dSdt[:, -1] = dedt   


    return dSdt.flatten()




# dim = 3
# kernel = cubicSplineKernel(dim, h=1.0)
# sys = SPHsystem(3,dim,kernel)

# rij, rij_norm  = sys.geom()

# x = np.array([0,2,4])
# y = np.array([0,3,1])
# z = np.array([0,0,0])

# r = np.column_stack((x,y,z))    # -> one particle at (x,y) = (0,0), one at (2,3) and one at (4,1)
# sys.S[:,:dim] = r


# print(r[0])
# print(r[1])
# print(r[2])



# 3D plotting

# x = np.linspace(-2, 5,100)
# y = np.linspace(-2, 5,100)
# X, Y = np.meshgrid(x, y)
# Z = np.zeros_like(X)
# Wtot = np.zeros_like(X)

# for r_i in r:
#     dx = X - r_i[0]
#     dy = Y - r_i[1]
#     dz = Z - r_i[2]
#     R  = np.sqrt(dx**2 + dy**2 + dz**2)
#     Wtot += kernel.W(R)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Wtot, cmap="viridis")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# plt.show()



# ================ testing gradW ==============

# # 1D
# N = 100
# dim = 1
# kernel = cubicSplineKernel(dim, h=1.0)
# sys = SPHsystem(N, dim, kernel)

# print(sys.S)

# sys.S[:,0] = np.linspace(0,10,N)

# plt.imshow(sys.S)
# #plt.show()

# r = sys.r
# print(r.shape)
# x = r[:,0]

# rij, rij_norm = sys.geom()
# dw = kernel.gradW(rij, rij_norm)

# # w = kernel.W(rij_norm)
# # plt.imshow(w)
# # plt.colorbar()
# # plt.show()

# plt.imshow(dw)
# plt.colorbar()
# plt.show()


# print(dw.shape)
# print(dw)


# #3D
# N = 100
# dim = 3
# kernel = cubicSplineKernel(dim,h=1.0)
# system = SPHsystem(N,dim,kernel)

# r = np.column_stack((np.linspace(0,10,N), np.linspace(0,10,N), np.linspace(0,10,N)))

# system.S[:,:dim] = r

# print(system.S)
# plt.imshow(system.S)
# plt.show()

# rij, rij_norm = system.geom()
# print(rij.shape)
# print(rij_norm.shape)

# print(rij_norm)
# W = kernel.W(rij_norm)
# print(W.shape)

# plt.imshow(W)
# plt.show()

# dw = kernel.gradW(rij, rij_norm)
# print(dw.shape)


# # test slicing
# N = 3
# dim = 3
# kernel = cubicSplineKernel(dim,h=1.0)
# sys = SPHsystem(N,dim,kernel)

# vx = np.array([1,2,3])
# vy = np.array([5,4,3])
# vz = np.array([0,0,0])

# v_vec = np.column_stack((vx,vy,vz))
# sys.S[:,dim:2*dim] = v_vec


# print("vx particle 0 = ", sys.v[0][0])
# print("vx particle 1 = ", sys.v[1][0])
# print("vx particle 2 = ", sys.v[2][0])

# print("vy particle 0 = ", sys.v[0][1])

# #print(sys.S)

# # v[i][j][0] v_xi - v_xj
# # v[i][j][1] v_yi - v_yj
# vij = sys.v[:,np.newaxis,:] - sys.v[np.newaxis,:,:]



# print("v_x0 - v_x1 = ", vij[0,1,0])




