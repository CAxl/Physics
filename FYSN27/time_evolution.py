import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
a = 1
hbar = 1
m = 1

def phi_n(n, x):
    return np.sqrt(2/a) * np.sin(n * np.pi * x / a)

def E_n(n):
    return (n**2 * np.pi**2 * hbar**2) / (2 * m * a**2)

def c_n(n, t):
    c_n0 = 1/np.sqrt(2)
    return c_n0 * np.exp(-1j * E_n(n) * t / hbar)

def psi_n(n, x, t):
    
    return phi_n(n, x) * np.exp(-1j * E_n(n) * t / hbar)

def psi_t(N, x, t):
    psi = np.zeros_like(x, dtype=complex)
    for n in range(1, N+1):
        psi += c_n(n, t) * phi_n(n, x)
    return psi

# Space grid
x = np.linspace(0, a, 1000)

# Precompute static φ₁ and φ₂
phi1 = phi_n(1, x)
phi2 = phi_n(2, x)

# Create animation figure
fig, ax = plt.subplots()

# Static shapes
ax.plot(x, phi1, 'b--', alpha=0.5, label='φ₁(x) (static)')
ax.plot(x, phi2, 'r--', alpha=0.5, label='φ₂(x) (static)')

# Animated lines
line_psi, = ax.plot([], [], 'k', lw=2, label='Re[ψ(x,t)]')
line_phi1t, = ax.plot([], [], 'b', alpha=0.4, label='Re[ψ₁(x,t)]')
line_phi2t, = ax.plot([], [], 'r', alpha=0.4, label='Re[ψ₂(x,t)]')

ax.set_xlim(0, a)
ax.set_ylim(-2, 2)
ax.set_xlabel("x")
ax.set_ylabel("Wavefunction")
ax.legend()

def init():
    line_psi.set_data([], [])
    line_phi1t.set_data([], [])
    line_phi2t.set_data([], [])
    return line_psi, line_phi1t, line_phi2t

def update(frame):
    t = frame * 0.005
    psi = psi_t(2, x, t)
    psi1 = psi_n(1, x, t)
    psi2 = psi_n(2, x, t)

    line_psi.set_data(x, np.real(psi))
    line_phi1t.set_data(x, np.real(psi1))
    line_phi2t.set_data(x, np.real(psi2))

    return line_psi, line_phi1t, line_phi2t

ani = FuncAnimation(fig, update, frames=1000, init_func=init,
                    blit=True, interval=20)
plt.show()





