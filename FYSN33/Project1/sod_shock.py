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
#print(x_L)

# right region
Nx_R = 80
dx_R = 0.0075
x_R = dx_R * np.arange(1, Nx_R+1)
#print(x_R)


x = np.concatenate([x_L,x_R])
N = len(x)
#print(x)
#print(N, "hopefully = 320 + 80 = 400")


# SPHsys setup
kernel = main.cubicSplineKernel(h=0.04) # calculate h later, see slides
system = main.SPHsystem(N, kernel)

# initialize statevector
system.m[:] = 0.001875

system.S[:,0] = x   # position
system.S[:,1] = 0.0 # velocity (v_R = v_L = 0)
system.density_summation()  # depends on spacing
system.S[x<=0, 3] = 2.5 # left region energy
system.S[x>0, 3] = 1.795 # right region energy


"""
rho_L approx 1, rho_R approx 0.25 (nice) 
(different at endpoints, makes physical sense)           
"""
# print(system.S[:,2])   

# print(system.S)

# # Looks correct, we have a density discontinuity at x=0
# plt.scatter(system.S[:,0], system.S[:,2])
# plt.xlim((-0.4, 0.4))
# plt.show()

# ========================================================

# ------------------- solver -----------------------------
t0 = 0.0
T = 0.2
Nsteps = 40

times = np.linspace(0,T,Nsteps) # should be 40 time steps (to reproduce slides)
print(times[1] - times[0])  # dt = 0.005

S0 = system.S.ravel()   # flatten S -> [x1, v1, rho1, e1, x2, v2,...]
NS = main.NavierStokes1D()

# for i in range(len(S0)):
#     print(S0[i])


sol = solve_ivp(
    fun= lambda t,y : main.RHS(t,y,system,NS),
    t_span=(t0,T),
    y0=S0,
    method="RK45",
    t_eval=times,
    rtol=1e-4,
    atol=1e-7
)

# --------------------------------------------------------


# ================== static plot rho vs x =========================
S_final = sol.y[:,-1].reshape(N,4)

# update the state vector
system.S[:] = S_final
# recompute the density! (is it sufficient to only do this at the end?)
system.density_summation()

# collect the results
x_final = system.S[:,0]
v_final = system.S[:,1]
rho_final = system.S[:,2]
e_final = system.S[:,3]

plt.figure()
plt.scatter(x_final,rho_final,s=8)
plt.xlim((-0.4,0.4))
plt.xlabel("x")
plt.ylabel("rho")
plt.title("SPH Sod shock - density(x)")
plt.grid()
plt.show()

plt.figure()
plt.scatter(x_final,v_final,s=8)
plt.xlim((-0.4,0.4))
plt.xlabel("x")
plt.ylabel("velocity, v")
plt.title("SPH Sod shock - velocity(x)")
plt.grid()
plt.show()


plt.figure()
plt.scatter(x_final,e_final,s=8)
plt.xlim((-0.4,0.4))
plt.xlabel("x")
plt.ylabel("energy, e")
plt.title("SPH Sod shock - energy(x)")
plt.grid()
plt.show()




# # ===================== Sod shock simulation (no viscosity) =======================
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
# ani.save("sod_density_profile.mp4", writer=writer)











