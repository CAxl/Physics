import numpy as np
import matplotlib.pyplot as plt
import time
import main
from matplotlib.pyplot import imshow


# # Broadcasting
# x = np.linspace(7, 45, 100)
# y = np.linspace(90, 5, 100)

# coly = y[:,np.newaxis] # shape as column vector

# # Broadcasting sum
# t0 = time.time()
# summ = x[:,np.newaxis] + y
# t1 = time.time()

# print("this took",  round(t1 - t0, 4), "s")

# # Double loop for matrix sum
# t0 = time.time()
# sumxy = np.zeros((len(x), len(y)))
# for i in range(len(x)):
#     for j in range(len(y)):
#         sumxy[i][j] = x[i] + y[j]        
# t1 = time.time()

# print("looping sum takes ", round(t1 - t0, 4), "seconds")


# print("=========================================================")
# print(summ)
# print("=========================================================")
# print(sumxy)


# testing N particles with kernels

N = 100
kernel = main.cubicSplineKernel(h=1.0)
sys = main.SPHsystem(N, kernel)
sys.S[:,0] = np.linspace(-10,10,N) # NOTE: could also initialize as sys.x == sys.S[:,0]

dx, r = sys.geom()
#print("dx matrix = ", dx)

W = kernel.W(r)
print("kernel W(r) = ", W)
# imshow(W)
# plt.colorbar()
# plt.show()

dW = kernel.gradW(dx, r)
# imshow(dW)
# plt.colorbar()
# plt.show()

x = sys.x

# for i in range(sys.N):
#     plt.plot(x, W[i,:], marker = 'o', label=f"kernel at x[{i}] = {x[i]:.1f}")
#     plt.axvline(x[i], linestyle='--', alpha = 0.5)

# plt.xlabel("absolute position, x")
# plt.ylabel("W(|x_i-x_j|)")
# plt.legend()
# plt.title("felt interaction for particle x_j due to particle x_i")
# plt.show()

# plot the 10:th particle's kernel and the 89:th particle's kernel
plt.plot(x, W[10,:], marker = 'o',label=f"kernel at x[{10}] = {x[10]:.1f}")
plt.plot(x, W[90,:], marker = 'o',label=f"kernel at x[{90}] = {x[90]:.1f}")
plt.axvline(x[10], linestyle = '--', alpha = 0.5)
plt.axvline(x[90], linestyle = '--', alpha = 0.5)
plt.xlabel("x")
plt.ylabel("W(r)")
plt.legend()
#plt.show()



# density summation test
sys.density_summation()
rho = sys.rho


print("energies = ", sys.e)

# pressure test
operator = main.NavierStokes1D()

pressure = operator.pressure(sys)
print("pressure = ", pressure)

# sanity check for state vector:
S = sys.S
print(S) # seems correct, position col, vel. col = \vec{0}, density, energy = 2.5 \forall i
print(S.flatten()) # -> [x_1, v_1, \rho_1, e_1, x_2, v_2, ...] (this good ?)









# def RK4_step(f, t, y, h):
#     k1 = f(t,y)
#     k2 = f(t + h/2, y + h/2 * k1)
#     k3 = f(t + h/2, y + h/2 * k2)
#     k4 = f(t + h, y + h * k3)
#     return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

# def integrate(system, NSequations, t0, T, dt):
#     t = t0
#     S = system.S.ravel()

#     history = []
#     times = []

#     while t < T:
#         history.append(S.reshape(system.N, 4).copy())
#         times.append(t)

#         S = RK4_step(lambda tt, SS: RHS(tt, SS, system, NSequations), t, S, dt)

#         t += dt

#     return np.array(times), np.array(history)


# N = 50
# x = np.linspace(0,1,N)

# kernel = cubicSplineKernel(h=0.04)
# system = SPHsystem(N, kernel)

# # initialize S with x-positions
# system.S[:,0] = x
# #initialize S with v = 0
# system.S[:,1] = 0.0
# system.S[:,3] = 1.0

# # larger energy to the right
# system.S[x<0.5, 3] = 2.5

# NS = NavierStokes1D()
# times, hist = integrate(system, NS, t0=0.0, T=0.2, dt=1e-4)

# for i in range(0, len(times), 5):
#     plt.clf()
#     plt.scatter(hist[i,:,0], np.zeros(N), c=hist[i,:,3])

#     plt.title(f"t = {times[i]:.4f}")
#     plt.xlabel("x")
#     plt.colorbar()
#     plt.pause(0.1)










