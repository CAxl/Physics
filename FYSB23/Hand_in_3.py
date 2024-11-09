import numpy as np
import matplotlib.pyplot as plt 

plt.rcParams['text.usetex'] = True

k_b = 1.380649 * 10**(-23)

t = np.linspace(0,30,1000)

Z = np.exp(-(1/t)) + np.exp(-(2/t)) + np.exp(-(4/t))

p0 = np.exp(-1/t)/Z
p1 = np.exp(-2/t)/Z
p2 = np.exp(-4/t)/Z

plt.plot(t, p0, label=r"$p_0$")
plt.plot(t,p1, label=r"$p_1$")
plt.plot(t,p2, label=r"$p_2$")
plt.xlabel("T")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.show()

