import numpy as np
import matplotlib.pyplot as plt


def F(x, mu, a):
    
    den = 1 + np.exp(a*(x-mu))

    return 1/den


x = np.linspace(0.1,10,2000)
mu = 5

plt.plot(x, F(x, mu, 0.5), label = "a = 0.5")
plt.plot(x, F(x, mu, 1), label = "a = 1")
plt.plot(x, F(x, mu, 2), label = "a = 2")
plt.plot(x, F(x, mu, 3), label = "a = 3")
plt.plot(x, F(x, mu, 4), label = "a = 4")
plt.plot(x, F(x, mu, 5), label = "a = 5")


plt.xlabel("x")
plt.ylabel("F(x)")

plt.legend()
plt.show()





def diff(func, x, h=1e-5, *args):
    """Central difference derivative of func(x, *args).
       x may be a numpy array or scalar. Vectorized.
    """
    return (func(x + h, *args) - func(x - h, *args)) / (2.0*h)


# grid and parameters
x = np.linspace(0.1, 10, 2000)
mu = 5.0
a = 3

# function values
Fvals = F(x, mu, a)
Fprime = diff(F, x, 1e-5, mu, a)
Fprime_analytic = -a * Fvals * (1 - Fvals)

# measure FWHM
absFp = np.abs(Fprime)
peak = absFp.max()
half = peak / 2
print(peak)
print(half)

indices = np.where(absFp >= half)[0]
print(indices)
x_right = x[indices[-1]]
x_left = x[indices[0]]
deltX = x_right - x_left # FWHM
print(deltX)

# theoretical FWHM
theoretical_fwhm = 4.0 * np.arccosh(np.sqrt(2.0)) / a # \propto 1/a

print(f"Measured numerical FWHM Δx ≈ {deltX:.6f}")
print(f"Theoretical FWHM Δx ≈ {theoretical_fwhm:.6f} ( = 3.5254909 / a )")


# plot
plt.figure(figsize=(8,4))
plt.plot(x, Fvals, label="F(x)")
plt.plot(x, Fprime, '--', label="dF/dx (numerical)")
plt.plot(x, Fprime_analytic, ':', label="dF/dx (analytic)")
plt.xlabel("x")
plt.hlines(-half, x_left, x_right, color = "black", linestyle = '--', label = "half max")
plt.legend()
plt.tight_layout()
plt.show()

