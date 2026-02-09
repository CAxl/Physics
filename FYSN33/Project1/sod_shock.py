import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter

import main


# ============== build the initial geometry: ===============

# left region
Nx_L = 320
dx_L = 0.001875
x_L = -dx_L * np.arange(Nx_L-1,-1,-1) # arange magic

# right region
Nx_R = 80
dx_R = 0.0075
x_R = dx_R * np.arange(1, Nx_R+1)

x = np.concatenate([x_L,x_R])
N = len(x)


# # SPHsys setup
kernel = main.cubicSplineKernel(h=0.01) # calculate h later, see slides
system = main.SPHsystem(N, kernel)

# initialize statevector
system.m[:] = 0.001875

system.S[:,0] = x   # position
system.S[:,1] = 0.0 # velocity (v_R = v_L = 0)
system.density_summation()  # depends on spacing
system.S[x<=0, 3] = 2.5 # left region energy
system.S[x>0, 3] = 1.795 # right region energy


# # ------------------- solver -----------------------------
t0 = 0.0
dt = 0.005
Nsteps = 40

times = np.linspace(0,dt*Nsteps,Nsteps) # should be 40 time steps (to reproduce slides)
print(times[1] - times[0])  # dt = 0.005

S0 = system.S.ravel()   # flatten S -> [x1, v1, rho1, e1, x2, v2,...]
NS = main.NavierStokes1D()


sol = solve_ivp(
    fun=lambda t, y: main.RHS(t, y, system, NS),
    t_span=(t0, dt*Nsteps),
    y0=S0,
    t_eval=times,
    method="RK45",
    max_step=dt,
    rtol=1e-4,
    atol=1e-7
)

# # ________________________ static plots ____________________________
# # trying to reproduce the plots in the slides...

k = -1  # last (40:th) time step
S_flat_k = sol.y[:,k]

S_k = S_flat_k.reshape(N,4)
system.S[:] = S_k
system.density_summation()

x_k = system.x
v_k = system.v
rho_k = system.rho
e_k = system.e
P_k = system.pressure()

print(f"{k}:th pressure vector", P_k)


print(f"S at time step = {k}")
print()
print(S_k)

plt.plot(x_k, rho_k)
plt.xlim((-0.4,0.4))
plt.xlabel("x")
plt.ylabel("rho")
plt.show()

plt.plot(x_k, P_k)
plt.xlim((-0.4,0.4))
plt.xlabel("x")
plt.ylabel("P")
plt.show()

plt.plot(x_k, v_k)
plt.xlabel("x")
plt.ylabel("v")
plt.xlim((-0.4,0.4))
plt.show()




# #===================== Sod shock simulation =======================
# fig, ax = plt.subplots(figsize=(7,4))

# scat = ax.scatter([],[],s=12)
# ax.set_xlim(-0.4,0.4)
# ax.set_ylim(0.0,1.2)

# ax.set_xlabel("x")
# ax.set_ylabel("rho(x)")

# def update(frame):
#     S = sol.y[:,frame].reshape(N,4) # sol.y is flattened

#     # update state vector every frame
#     system.S[:] = S

#     # recompute density EACH FRAME (!!! this HAS to be done when putting this into class)
#     system.density_summation()

#     # collect position and density
#     x = system.x
#     rho = system.rho

#     scat.set_offsets(np.column_stack((x,rho)))

#     ax.set_title(f"Density evolution, t = {sol.t[frame]:.4f}")

#     return scat,

# ani = FuncAnimation(fig, update, frames=len(sol.t),interval=50,blit=True)
# writer = FFMpegWriter(fps=20,bitrate=1800)
# ani.save("./results/sod_density_debugging.mp4", writer=writer)



# pressure vs x

# fig, ax = plt.subplots(figsize=(7,4))

# line, = ax.plot([],[],lw=2)
# ax.set_xlim(-0.4,0.4)
# ax.set_ylim(0.0,1.2)

# ax.set_xlabel("x")
# ax.set_ylabel("P(x)")

# def update(frame):
#     S_flat = sol.y[:, frame]
#     S = S_flat.reshape(N,4)

#     x = S[:,0]
#     rho = S[:,2]

#     line.set_data(x,rho)

#     return (line,)

# ani = FuncAnimation(fig, update, frames=len(sol.t),interval=50,blit=True)
# writer = FFMpegWriter(fps=20,bitrate=1800)
# ani.save("./results/sod_pressure_profilec.mp4", writer=writer)









