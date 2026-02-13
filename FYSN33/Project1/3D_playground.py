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



# # two planet test
# kernel = threeD.cubicSplineKernel(3,h=1e7)
# system = threeD.SPHsystem(2,3,kernel)

# system.S[0,:3] = [-1,0,0]
# system.S[1,:3] = [ 1,0,0]

# system.m = np.array([1,1])

# NS = threeD.NSequations(selfgrav_flag=True)

# print("NS.selfgrav(sys) = ", NS.selfgravity(system))



# ======================== Planet collision =====================

# ========================= read .dat file ======================

data = np.loadtxt("Planet300.dat")
#print(data.shape)

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
#print(N) # 301 planets lmao

kernel = threeD.cubicSplineKernel(dim, h = 5*1e6)
system = threeD.SPHsystem(N, dim, kernel)

gamma = system.gamma

system.S[:,0:3] = r
system.S[:,3:6] = v
system.S[:,-2] = rho
system.S[:,-1] = P / ((gamma - 1) * rho) # (e ??)
system.m = m    

# ------------- add spin --------------
T_spin = 8.5e3
threeD.add_spin(system, np.arange(system.N), T_spin)

# #give random velocities (no self gravity yet)
# system.S[:,3:6] += 1e4 * np.random.randn(N,3)


# ==================== solver =========================

"""
PARAMETER "SCAN":
___________________
Slow attraction (oscillating):
G = 10^{-11}
h = 1e7
Nsteps = 300
dt = 20
=> t[-1] = 6000
looks like gas cloud, doesnt pull together completely
___________________
Same as above but video lasts longer
G = 10^{-11}
h = 1e7
Nsteps = 800
dt = 20
=> t[-1] = 16000

__________________
G = 10^{-11}
h = 2e7
Nsteps = 800
dt = 20
=> t[-1] = 16000

diverges more before contracting (still gas-like)

_____________________
G = 10^{-11}
h = 5*1e6
Nsteps = 600
dt = 20
=> t[-1] = 12000

Fast inward contraction but outer shell of planets slower, looks a bit funky
(best sofar)
_____________________
G = 1e-11
h = 5*1e6
Nsteps = 2000
dt = 2
=>t[-1] = 4000

(added colormap for density here)
1:06 long video, looks cool at contracts and then oscillates a bit
becomes completely yellow when fully contracted... (i.e. max density on colormap)
____________________
(added spin here)

G = 1e-11
h = 5*1e6
Nsteps = 2000
dt = 10
=>t[-1] = 20000
T_spin = 8.5e6

similar to above, but spin is not noticeable
____________________
G = 1e-11
h = 5*1e6
Nsteps = 2000
dt = 10
=>t[-1] = 20000
T_spin = 8.5e3

(very succesfull video :) shows spinning and disk formation)
(best result sofar for single "planet/galaxy")
____________________
"""


t0 = 0.0
dt = 10
Nsteps = 2000

times = np.linspace(0,dt*Nsteps,Nsteps)
print(times[-1])


S0 = system.S.flatten()
NS = threeD.NSequations(selfgrav_flag=True)


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


# # debugging static plot 
# S_initial = sol.y[:,0].reshape(N, 2*dim+2)
# S_final   = sol.y[:,-1].reshape(N, 2*dim+2)

# disp = np.linalg.norm(S_final[:,0:3] - S_initial[:,0:3], axis=1)

# print("Max displacement:", np.max(disp))    # displacement smaller than radii when dt =0.01 N=100

# # debugging error message: Ivalid value in np.sqrt((self.gamma - 1) * self.e)
# print("Min energy:", np.min(S_final[:, -1]))
# # using dt = 20, Nsteps = 300 I get massively negative self energies (Min energy: -12938018.008582849)





# ================ plotting 3D ==========================

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')

# initial state from solver
S_init = sol.y[:,0].reshape(N, 2*dim + 2)
system.S[:] = S_init
system.density_summation()

rho_init = system.rho

sc = ax.scatter(
    S_init[:,0],
    S_init[:,1],
    S_init[:,2],
    c=rho_init,
    cmap='plasma',
    s=5
)

cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label("Density")

# set color scale from initial density
vmin = np.min(rho_init)
vmax = np.max(rho_init)
sc.set_clim(vmin, vmax)

# fix axes so they don’t rescale every frame
lim = 1.2 * np.max(np.abs(S_init[:,0:3]))
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


# ------- update function ----------
def update(frame):
    S = sol.y[:,frame].reshape(N, 2*dim + 2)
    system.S[:] = S

    system.density_summation()
    rho = system.rho

    x = S[:,0]
    y = S[:,1]
    z = S[:,2]

    sc._offsets3d = (x, y, z)
    sc.set_array(rho)

    ax.set_title(f"t = {sol.t[frame]:.3f}")
    return sc,


# ---------- animation ----------
ani = FuncAnimation(
    fig,
    update,
    frames=len(sol.t),
    interval=50,
    blit=False
)

writer = FFMpegWriter(fps=30)
ani.save("./results/planet300_testing.mp4", writer=writer)

















