
from sympy import roots
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


# h, k, g, m, U, E = sp.symbols("h k g m U E")

# Expr = ( (h**2*k**2) / (2*m) - E) * ( h**2*(k-g)**2 / (2*m) - E) - U**2

# smpl = sp.simplify(Expr)
# #print(smpl)

# roots = sp.roots(Expr, E)
# #print(roots)
# print(sp.simplify(roots))

h = 1
g = 1
U = 0
m = 1

kvals = np.linspace(-1,1,1000)
plt.plot(kvals,[h**2*k**2/(2*m) for k in kvals])
plt.plot(kvals,[h**2*(k-g)**2/(2*m) for k in kvals])
plt.show()


def E1(k):
    U = 0
    h = 1
    g = 1
    m = 1
    E1 = h**2 * (k**2 + (k-g)**2) / (4*m) + np.sqrt((h**2*k**2 - h**2*(k-g)**2)**2 / (4*(2*m)**2) + U**2)
    return E1

def E2(k):
    U = 0
    h = 1
    g = 1
    m = 1
    E2 = h**2 * (k**2 + (k-g)**2) / (4*m) - np.sqrt((h**2*k**2 - h**2*(k-g)**2)**2 / (4*(2*m)**2) + U**2)
    return E2
    

plt.plot(kvals,E1(kvals), label="E_+")
plt.plot(kvals, E2(kvals), label = "E_-")
plt.legend()
plt.show()


