import numpy as np
import matplotlib.pyplot as plt 


def C(x):
    return -(1/x**2)*(np.exp(-1/x)) / (1-np.exp(-1/x))**2

x = np.linspace(0, 3, 10000)
plt.plot(x,C(x))
plt.show()


k_B = 1.380649 * 10**(-23)
p = 101325
V = 4*np.pi*(0.11**2) / 3

T_room = 298.15
T_winter = 268.15
T_Qatar = 273.15 + 40

def N(T):
    return p*V /(k_B*T)

print("N(T_room) = ", N(T_room))
print("N(T_winter) = ", N(T_winter))
print("N(T_Qatar) = ", N(T_Qatar))
print()

print("E_room = ", (5/2)*N(T_room)*k_B*T_room)
print("E_winter = ", (5/2)*N(T_winter)*k_B*T_winter)
print("E_room = ", (5/2)*N(T_Qatar)*k_B*T_Qatar)



g = 9.82
k_B = 1.380649 * 10**(-23)
p_0 = 101325
V = 4*np.pi*(0.11**2) / 3
m = 4.65 * 10**-26
h= 2200
T = 273.15 + 25
A = 6.023 * 10**23

P = p_0 * np.exp(-m*g*h /(k_B*T))
print("P = ", P, "Pa")
print(28 / A)

N = P*V/(k_B*T)
print(N/A)

print("N = ", N)
print("E = ", (5/2)*N*k_B*T)