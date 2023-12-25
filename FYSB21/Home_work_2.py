import numpy as np
import scipy as sci 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

m1=m2=1
phi1 = 0.1
dphi1 = 0
phi2 = 0
dphi2 = 0
t = 0
g = 9.81
k = 1
L = 1
mu = 2

A = 0.05
B = 0.05

n = 10
tvals = np.linspace(0, 2*np.pi*n, 1000)

def phi1(t):
    term1 = A * np.cos(t*np.sqrt(g/L))
    term2 = B * np.cos(t*np.sqrt(g/L + k*mu)) * m2
    return term1 + term2

def phi2(t):
    term1 = A * np.cos(t*np.sqrt(g/L))
    term2 = B * np.cos(t*np.sqrt(g/L + k*mu)) * (-m1)
    return term1 + term2

plt.plot(tvals, phi1(tvals), label=r"$\phi_1(t)$")
plt.plot(tvals, phi2(tvals), label=r"$\phi_2(t)$")
plt.xlabel("Time t [s]")
plt.ylabel("Amplitude [m]")
plt.tick_params(direction = "in", right = "True", top = "True")
plt.legend()
plt.show()


plt.plot(tvals, phi1(tvals) + phi2(tvals))
plt.xlabel("Time t [s]")
plt.ylabel(r"$\phi_1(t) + \phi_2(t)$")
plt.tick_params(direction = "in", right = "True", top = "True")
plt.show()



# task 3

# part (i)

C = 0.1
h0 = 0
k0 = 2*np.pi / 5
alpha = k0 / 5

def f(x):
    return h0 + C*np.exp(-alpha*np.abs(x)) * np.cos(k0*x)

xvals = np.linspace(-50,50,1000)

plt.plot(xvals, f(xvals))
plt.grid()
plt.tick_params(direction = "in", right = "True", top = "True")
plt.ylabel("Amplitude [m]")
plt.xlabel("x [m]")
plt.title("Spatial profile of water wave at time t=0")
plt.show()

def f_tilde(k):
    integrand = lambda x: C*np.exp(-alpha*np.abs(x))*np.cos(k0*x) * np.exp(-1j*k*x)
    return h0 + sci.integrate.quad(integrand, -np.inf, np.inf)[0]

f_tilde = np.vectorize(f_tilde)

kvals = np.linspace(0,5,100)

plt.plot(kvals, f_tilde(kvals))
plt.xlabel("k [m$^{-1}$]")
plt.ylabel("h(k,0)")
plt.axhline(y=0.2, color="red", linestyle="--", label="max{h(k)}/2")
plt.axvline(x=k0 + 1/4, color="black",linestyle="--", label = "k$_0$ + 1/4")
plt.legend()
plt.grid()
plt.tick_params(direction = "in", right = "True", top = "True")
plt.show()



# Part (ii)

h0 = 0.25
g = 9.81
h0_d = 5

def h(x,t): # approx |k| = k
    integrand = lambda k: (1/((k-k0)**2 + alpha**2))*np.cos(k*x - np.sqrt(g*h0)*k*t)
    return h0 + alpha*C*(1/np.pi)*sci.integrate.quad(integrand,-np.inf, np.inf)[0]

def habs(x,t): # no approx 
    integrand = lambda k: (1/((k-k0)**2 + alpha**2))*np.cos(k*x - np.sqrt(g*h0)*np.abs(k)*t)
    return h0 + alpha*C*(1/np.pi)*sci.integrate.quad(integrand,-np.inf, np.inf)[0]

h = np.vectorize(h)
habs = np.vectorize(habs)

def hd(x,t):    # deep water wave
    integrand = lambda k: (1/((k-k0)**2 + alpha**2))*np.cos(k*x - np.sqrt(g*np.abs(k))*t)
    return h0_d + alpha*C*(1/np.pi)*sci.integrate.quad(integrand,-np.inf, np.inf)[0]

hd = np.vectorize(hd)

t1 = 0
t2 = 10
t3 = 20
t4 = 30

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.plot(xvals, h(xvals, t1), label = f"$h(x, t={t1}$s)")   # approx |k|=k
ax2.plot(xvals, h(xvals, t2), label = f"$h(x, t={t2}$s)")
ax3.plot(xvals, h(xvals, t3), label = f"$h(x, t={t3}$s)")
ax4.plot(xvals, h(xvals, t4), label = f"$h(x, t={t4}$s)")
ax1.plot(xvals, habs(xvals, t1), alpha=0.8)    # no approx
ax2.plot(xvals, habs(xvals, t2), alpha=0.8)
ax3.plot(xvals, habs(xvals, t3), alpha=0.8)
ax4.plot(xvals, habs(xvals, t4), alpha=0.8)
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax1.tick_params(direction = "in", right = "True", top = "True")
ax2.tick_params(direction = "in", right = "True", top = "True")
ax3.tick_params(direction = "in", right = "True", top = "True")
ax4.tick_params(direction = "in", right = "True", top = "True")
ax1.set_ylabel("Amplitude [m]")
ax2.set_ylabel("Amplitude [m]")
ax3.set_ylabel("Amplitude [m]")
ax4.set_ylabel("Amplitude [m]")
ax1.set_xlabel("x [m]")
ax2.set_xlabel("x [m]")
ax3.set_xlabel("x [m]")
ax4.set_xlabel("x [m]")
fig.suptitle("Time evolution for shallow water wave")
plt.show()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.plot(xvals, hd(xvals, t1), label=f"$h(x,t={t1}$s)")
ax2.plot(xvals, hd(xvals, t2), label=f"$h(x,t={t2}$s)")
ax3.plot(xvals, hd(xvals, t3), label=f"$h(x,t={t3}$s)")
ax4.plot(xvals, hd(xvals, t4), label=f"$h(x,t={t4}$s)")
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax1.tick_params(direction = "in", right = "True", top = "True")
ax2.tick_params(direction = "in", right = "True", top = "True")
ax3.tick_params(direction = "in", right = "True", top = "True")
ax4.tick_params(direction = "in", right = "True", top = "True")
ax1.set_ylabel("Amplitude [m]")
ax2.set_ylabel("Amplitude [m]")
ax3.set_ylabel("Amplitude [m]")
ax4.set_ylabel("Amplitude [m]")
ax1.set_xlabel("x [m]")
ax2.set_xlabel("x [m]")
ax3.set_xlabel("x [m]")
ax4.set_xlabel("x [m]")
fig.suptitle("Time evolution for deep water wave")

plt.show()

    
