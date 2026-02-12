import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter

import threeD


# # =================== SOD SHOCK =======================
# # ============ build Sod shock geometry ===============

# dim = 1

# # left region
# Nx_L = 320
# dx_L = 0.001875
# x_L = -dx_L * np.arange(Nx_L-1,-1,-1) # arange magic

# # right region
# Nx_R = 80
# dx_R = 0.0075 # particle separation
# x_R = dx_R * np.arange(1, Nx_R + 1)

# x = np.concatenate([x_L,x_R])
# N = len(x)


# # SPH sys
# kernel = threeD.cubicSplineKernel(dim,h=0.01)
# sys = threeD.SPHsystem(N, dim, kernel)


# sys.m[:] = 0.001875
# sys.S[:,0] = x
# #print(sys.r)
# sys.S[:,1] = 0.0
# sys.density_summation()
# sys.S[x<=0, 3] = 2.5    # left region energy
# sys.S[x>0 , 3] = 1.795  # right region energy
# print(sys.S)


# # ============= solver ============

# t0 = 0.0
# dt = 0.005
# Nsteps = 40

# times = np.linspace(0,dt*Nsteps,Nsteps)
# print(times)


# S0 = sys.S.flatten()
# NS = threeD.NSequations()


# sol = solve_ivp(
#     fun=lambda t, y: threeD.RHS(t, y, sys, NS),
#     t_span=(t0, dt*Nsteps),
#     y0=S0,
#     t_eval=times,
#     method="RK45",
#     max_step=dt,
#     rtol=1e-4,
#     atol=1e-7
# )


# # ========================== static plots =======================
# k = -1  # last (40:th) time step
# S_flat_k = sol.y[:,k]

# S_k = S_flat_k.reshape(N,4)
# sys.S[:] = S_k
# sys.density_summation()

# x_k = sys.r
# v_k = sys.v
# rho_k = sys.rho
# e_k = sys.e
# P_k = sys.pressure()

# plt.plot(x_k, rho_k)
# plt.xlim((-0.4,0.4))
# plt.xlabel("x [m]")
# plt.ylabel("Density [kg/m³]")
# plt.grid()
# plt.show()

# plt.plot(x_k, P_k)
# plt.xlim((-0.4,0.4))
# plt.xlabel("x [m]")
# plt.ylabel("Pressure [N/m²]")
# plt.show()

# plt.plot(x_k, v_k)
# plt.xlabel("x [m]")
# plt.ylabel("Velocity [m/s]")
# plt.xlim((-0.4,0.4))
# plt.show()




# ======================== Planet collision =====================

# ========================= read .dat file ======================

data = np.loadtxt("Planet300.dat")
print(data.shape)

x = data[:,0]
y = data[:,1]
z = data[:,2]

r = data[:, 0:3]
v = data[:, 3:6]
m = data[:, 6]
rho = data[:, 7]
P = data[:, 8]

# print(r)
# print(P)

# ======================= SPH system setup =================

dim = 3
N = data.shape[0]
print(N) # 301 planets lmao

kernel = threeD.cubicSplineKernel(dim, h = 10e7)
system = threeD.SPHsystem(N, dim, kernel)

gamma = system.gamma

system.S[:,0:3] = r
system.S[:,3:6] = v
system.S[:,-2] = rho
system.S[:,-1] = P / ((gamma - 1) * rho) # (??)


# give random velocities (no self gravity yet)
#system.S[:,3:6] += 1e4 * np.random.randn(N,3)


# ==================== solver =========================

t0 = 0.0
dt = 20
Nsteps = 300

times = np.linspace(0,dt*Nsteps,Nsteps)
print(times[-1])


S0 = system.S.flatten()
NS = threeD.NSequations()


sol = solve_ivp(
    fun=lambda t, y: threeD.RHS(t, y, system, NS),
    t_span=(t0, dt*Nsteps),
    y0=S0,
    t_eval=times,
    method="RK45",
    max_step=dt,
    rtol=1e-4,
    atol=1e-7
)


# debugging static plot 
S_initial = sol.y[:,0].reshape(N, 2*dim+2)
S_final   = sol.y[:,-1].reshape(N, 2*dim+2)

disp = np.linalg.norm(S_final[:,0:3] - S_initial[:,0:3], axis=1)

print("Max displacement:", np.max(disp))    # displacement smaller than radii when dt =0.01 N=100

# debugging error message: Ivalid value in np.sqrt((self.gamma - 1) * self.e)
print("Min energy:", np.min(S_final[:, -1]))






# # ================ plotting 3D ==========================

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(projection='3d')

# # initial state from solver
# S_init = sol.y[:,0].reshape(N, 2*dim + 2)
# sc = ax.scatter(S_init[:,0], S_init[:,1], S_init[:,2], s=5)

# # optional: fix axes so they don’t rescale every frame
# lim = 1.2 * np.max(np.abs(S_init[:,0:3]))
# ax.set_xlim(-lim, lim)
# ax.set_ylim(-lim, lim)
# ax.set_zlim(-lim, lim)

# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")


# # ------- update function ----------
# def update(frame):
#     S = sol.y[:,frame].reshape(N, 2*dim + 2)
#     x = S[:,0]
#     y = S[:,1]
#     z = S[:,2]

#     sc._offsets3d = (x, y, z)
#     ax.set_title(f"t = {sol.t[frame]:.3f}")
#     return sc,


# # ---------- animation ----------
# ani = FuncAnimation(
#     fig,
#     update,
#     frames=len(sol.t),
#     interval=50,
#     blit=False
# )

# writer = FFMpegWriter(fps=30)
# ani.save("./results/planet300.mp4", writer=writer)

















