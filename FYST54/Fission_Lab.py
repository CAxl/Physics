import numpy as np
import matplotlib.pyplot as plt

def KE_viola(Z,A):
    return 0.1189 * (Z**2) /(A**(1/3)) + 7.3

def KE(Z1, Z2, R):
    return Z1*Z2*1.4399764 / R

Pt_viola = KE_viola(78, 180)
Pt_ke = KE(53, 26, 16.11)

print("Platinum (Z = 78, N = 102)")
print("KE_viola = ", Pt_viola)
print("KE = ", Pt_ke)

print()

Pb_viola = KE_viola(82, 208)
Pb_ke = KE(52,30,15.75)
print("Lead (Z = 82, N = 126)")
print("KE_viola = ", Pb_viola)
print("KE = ", Pb_ke)

print()

U_viola = KE_viola(92, 236)
U_ke = KE(51, 41, 16.15)
print("Uranium (Z = 92, N = 144)")
print("KE_viola = ", U_viola)
print("KE = ", U_ke)


# Pt, Pb, U
A_h = [121, 131, 130]
A_l = [59, 77, 106]


A_range = np.arange(170, 240, 1)
print(A_range)

plt.scatter(A_range, A_h, "x")
plt.scatter(A_range, A_l, "red", "x")
plt.show()
