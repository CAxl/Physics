import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import threeD


# =================== SOD SHOCK =======================
# ============ build Sod shock geometry ===============

dim = 1

# left region
Nx_L = 320
dx_L = 0.001875
x_L = -dx_L * np.arange(Nx_L-1,-1,-1) # arange magic

# right region
Nx_R = 80
dx_R = 0.0075 # particle separation
x_R = dx_R * np.arange(1, Nx_R + 1)

x = np.concatenate([x_L,x_R])
N = len(x)


# SPH sys
kernel = threeD.cubicSplineKernel(dim,h=0.01)
sys = threeD.SPHsystem(N, dim, kernel)


sys.m[:] = 0.001875
sys.S[:,0] = x
#print(sys.r)
sys.S[:,1] = 0.0
sys.density_summation()
sys.S[x<=0, 3] = 2.5    # left region energy
sys.S[x>0 , 3] = 1.795  # right region energy
print(sys.S)


# ============= solver ============

t0 = 0.0
dt = 0.005
Nsteps = 40

times = np.linspace(0,dt*Nsteps,Nsteps)
print(times)


S0 = sys.S.flatten()
NS = threeD.NSequations()


sol = solve_ivp(
    fun=lambda t, y: threeD.RHS(t, y, sys, NS),
    t_span=(t0, dt*Nsteps),
    y0=S0,
    t_eval=times,
    method="RK45",
    max_step=dt,
    rtol=1e-4,
    atol=1e-7
)


# ========================== static plots =======================
k = -1  # last (40:th) time step
S_flat_k = sol.y[:,k]

S_k = S_flat_k.reshape(N,4)
sys.S[:] = S_k
sys.density_summation()

x_k = sys.r
v_k = sys.v
rho_k = sys.rho
e_k = sys.e
P_k = sys.pressure()

plt.plot(x_k, rho_k)
plt.xlim((-0.4,0.4))
plt.xlabel("x [m]")
plt.ylabel("Density [kg/m³]")
plt.grid()
plt.show()

plt.plot(x_k, P_k)
plt.xlim((-0.4,0.4))
plt.xlabel("x [m]")
plt.ylabel("Pressure [N/m²]")
plt.show()

plt.plot(x_k, v_k)
plt.xlabel("x [m]")
plt.ylabel("Velocity [m/s]")
plt.xlim((-0.4,0.4))
plt.show()





