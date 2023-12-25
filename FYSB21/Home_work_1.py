import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['text.usetex'] = True

# Task 1

gamma = 6.3
mu = 0.5
l = 1
m = 0.2
phi_a = 0.2

t = np.linspace(0, 4, 500)

def phi(t):
    return -phi_a * np.exp(-gamma*t/2)

plt.plot(t, phi(t), "r")
plt.axhline(0)
plt.xlabel("time, t")
plt.ylabel(r"$\phi(t)$")
plt.show()



# Task 2

w_0 = 2 * np.pi * 440   # pure 'A'
s_0 = 1
T = 1

def s(w):   # fourier transform of s(t)
    R = (s_0/2) * (((np.exp(1j * (w - w_0) * T) - 1) / (w - w_0)) - 
                   ((np.exp(1j * (w + w_0) * T) - 1) / (w + w_0)))
    return R

def s_mod(w):  # modulus squared
    return s(w) * np.conj(s(w))

X = np.linspace(w_0 - 6*np.pi, w_0 + 6*np.pi, 500)  
dev1 = w_0 + 2*np.pi
dev2 = w_0 - 2*np.pi

plt.plot(X, s_mod(X))
plt.vlines([dev1, dev2], 0.25, 0, "r", label = r'w\textsubscript{0} $\pm$ 2$\pi$')
plt.vlines(w_0, 0.25, 0, "g", label = r'w\textsubscript{0} = ' + str(w_0.__round__(2)))
plt.xlabel(r"Angular frequecy, w")
plt.ylabel(r"Amplitude, s(w)")
plt.legend()
plt.show()

# plt.plot(X,s(X))
# plt.show()




# Task 3

T = 1
Kvals = np.linspace(0, 2.5, 500)
tau = T / 30

def p(t):   # piecewise defined p(t)
    return np.piecewise(t, [(t%T >= 0) & (t%T <= 2*np.pi), (t%T > 2*np.pi * (tau)) & (t%T < T)], [lambda t: np.sin((t%T)/tau), 0])

def p_k(j, t):
    p = 0
    for i in range(-j, j+1):
        w_j = (i*2*np.pi)/T
        a = 1/(2*T) * ( (np.exp(1j * (w_j - (1/tau)) * 2*np.pi*tau) - 1) / (w_j - (1/tau)) - 
                        (np.exp(1j * (w_j + (1/tau)) * 2*np.pi*tau) - 1) / (w_j + (1/tau)) )
        p += a * np.exp(-1j * w_j * t)
    return p



plt.plot(Kvals, p(Kvals), label=r"$p(t)$")
# plt.plot(K, p_k(0, K), label=r"$p_0(t)$")
# plt.plot(K, p_k(1, K), label=r"$p_1(t)$")
# plt.plot(K, p_k(5, K), label=r"$p_5(t)$")
plt.plot(Kvals, p_k(11, Kvals), label=r"$p_{11}(t)$", alpha = 0.5)
plt.legend()
plt.xlabel(r"time, t")
plt.ylabel(r"Amlpitude")
plt.show()


jvals = np.arange(0,21)
yvals = [p_k(jvals[0],k) for k in Kvals]

fig, ax = plt.subplots()
line = ax.plot(Kvals, yvals, color="tab:orange", label=r"$\Sigma a_k e^{-i\omega_k t}$ for k = 0")[0]
plt.plot(Kvals, p(Kvals), alpha=0.6, label=r"$f(t)$")
L = ax.legend(loc=1)
ax.tick_params(direction = "in", top="True", right="True")

def update(frame):
    j = jvals[frame]
    yvals = [p_k(j, K) for K in Kvals]
    line.set_ydata(yvals)
    lab = r"$\Sigma a_k e^{-i\omega_k t}$ for k = " +str(j)
    L.get_texts()[0].set_text(lab)
    return line

ani = animation.FuncAnimation(fig = fig, func = update, frames = 21, interval = 500)

# plt.show()
writergif = animation.PillowWriter(fps = 2)
ani.save("Fourier.gif", writer = writergif)
