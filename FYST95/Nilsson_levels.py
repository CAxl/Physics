import numpy as np
import matplotlib.pyplot as plt

A = 28
hbaromega0 = 41 * pow(A,-1/3) # MeV
print(hbaromega0)
mu = 5


# N = 2

def E10(delta):
	return hbaromega0 * ((5/2) - (2/3)*delta)

def E01(delta):
	return hbaromega0 * ((5/2) + (1/3)*delta)


def block1(delta):
	arr = np.array([[E10(delta), mu/np.sqrt(2)],
					[mu/np.sqrt(2), E01(delta) + mu/2]])
	
	return arr


def block2(delta):
	arr = np.array([[E10(delta), -mu/np.sqrt(2)],
					[-mu/np.sqrt(2), E01(delta) + mu/2]])
	
	return arr

# deformation range
deltas = np.linspace(-1, 1, 1000)

# storage
E_p32 = []   # \Omega = \pm3/2
E_m32 = []
E_p12_1 = [] # \Omega = +1/2
E_p12_2 = []
E_m12_1 = [] # \Omega = -1/2
E_m12_2 = []

for d in deltas:
	# Omega = \pm 3/2
	E_p32.append((E01(d) - mu/2) / hbaromega0) 
	E_m32.append((E01(d) - mu/2) / hbaromega0) 

	# Omega \pm 1/2 blocks
	e1 = np.linalg.eigvalsh(block1(d))
	e2 = np.linalg.eigvalsh(block2(d))

	E_p12_1.append(e1[0] / hbaromega0)
	E_p12_2.append(e1[1] / hbaromega0)
	E_m12_1.append(e2[0] / hbaromega0)
	E_m12_2.append(e2[1] / hbaromega0)

plt.figure(figsize=(7,9))

plt.plot(deltas, E_p32, 'k--', label=r'$\Omega=\pm 3/2$')
plt.plot(deltas, E_p12_1, 'b')
plt.plot(deltas, E_p12_2, 'b', label=r'$\Omega=+1/2$')
plt.plot(deltas, E_m12_1, 'r--')
plt.plot(deltas, E_m12_2, 'r--', label=r'$\Omega=-1/2$')





# N = 2

def E20(delta):
	return hbaromega0 * ((7/2)  - (4/3)*delta)

def E02(delta):
	return hbaromega0 * ((7/2) + (2/3)*delta)

def E11(delta):
	return hbaromega0 * ((7/2) - (1/3)*delta)


# def block21(delta): # Omega = +3/2
# 	arr = np.array([[E02(delta) + mu, 0],
# 					[0, E11(delta) - mu/2]])
# 	return arr

def block22(delta): # Omega = +1/2
	arr = np.array([[E02(delta), -mu/np.sqrt(2), 0],
					[-mu/np.sqrt(2), E11(delta) + mu/2, mu],
					[0, mu, E20(delta)]])
	return arr

def block23(delta): # Omega = -1/2
	arr = np.array([[E02(delta), -mu/np.sqrt(2), 0],
					[-mu/np.sqrt(2), E11(delta) + mu/2, -mu],
					[0, -mu, E20(delta)]])
	return arr

def block24(delta):
	arr = np.array([[E02(delta) + mu, -mu],
					[-mu, E11(delta) - mu/2]])
	return arr


# =========================
# N = 2 Nilsson plotter
# =========================

# storage
E2_p52 = []
E2_m52 = []

E2_p32_1 = []
E2_p32_2 = []

E2_m32_1 = []
E2_m32_2 = []

E2_p12_1 = []
E2_p12_2 = []
E2_p12_3 = []

E2_m12_1 = []
E2_m12_2 = []
E2_m12_3 = []

for d in deltas:

	# Omega = ±5/2 (1×1)
	E2_p52.append((E02(d) - mu) / hbaromega0)
	E2_m52.append((E02(d) - mu) / hbaromega0)

	# Omega = +3/2 (2×2)	# this gives something weird!
	# e_p32 = np.linalg.eigvalsh(block21(d))
	# E2_p32_1.append(e_p32[0] / hbaromega0)
	# E2_p32_2.append(e_p32[1] / hbaromega0)

	# Omega = -3/2 (2×2)
	e_m32 = np.linalg.eigvalsh(block24(d))
	E2_m32_1.append(e_m32[0] / hbaromega0)
	E2_m32_2.append(e_m32[1] / hbaromega0)

	# Omega = +1/2 (3×3)
	e_p12 = np.linalg.eigvalsh(block22(d))
	E2_p12_1.append(e_p12[0] / hbaromega0)
	E2_p12_2.append(e_p12[1] / hbaromega0)
	E2_p12_3.append(e_p12[2] / hbaromega0)

	# Omega = -1/2 (3×3)
	e_m12 = np.linalg.eigvalsh(block23(d))
	E2_m12_1.append(e_m12[0] / hbaromega0)
	E2_m12_2.append(e_m12[1] / hbaromega0)
	E2_m12_3.append(e_m12[2] / hbaromega0)

# Omega = ±5/2 (1×1)
E2_p52 = np.array(E2_p52)
E2_p52 = np.array(E2_m52)

# Omega = -3/2 (2×2)
E2_m32_1 = np.array(E2_m32_1)
E2_m32_2 = np.array(E2_m32_2)

# Omega = +1/2 (3×3)
E2_p12_1 = np.array(E2_p12_1)
E2_p12_2 = np.array(E2_p12_2)
E2_p12_3 = np.array(E2_p12_3)

# Omega = -1/2 (3×3)
E2_m12_1 = np.array(E2_m12_1)
E2_m12_2 = np.array(E2_m12_2)
E2_m12_3 = np.array(E2_m12_3)



energy_offset = 2	# arbitrary now -> why are they not off-set automatically???

# \pm5/2
plt.plot(deltas, E2_p52 + energy_offset, 'k-.', label=r'$\Omega=\pm5/2$')

# +3/2
# plt.plot(deltas, E2_p32_1, 'b')
# plt.plot(deltas, E2_p32_2, 'b', label=r'$\Omega=+3/2$')

# −3/2
plt.plot(deltas, E2_m32_1 + energy_offset, 'r--')
plt.plot(deltas, E2_m32_2 + energy_offset, 'r--', label=r'$\Omega=-3/2$')

# +1/2
plt.plot(deltas, E2_p12_1 + energy_offset, 'g')
plt.plot(deltas, E2_p12_2 + energy_offset, 'g')
plt.plot(deltas, E2_p12_3 + energy_offset, 'g', label=r'$\Omega=+1/2$')

# −1/2
plt.plot(deltas, E2_m12_1 + energy_offset, 'm:')
plt.plot(deltas, E2_m12_2 + energy_offset, 'm:')
plt.plot(deltas, E2_m12_3 + energy_offset, 'm:', label=r'$\Omega=-1/2$')


plt.xlabel(r'$\delta$')
plt.ylabel(r'$E/\hbar\omega_0$')
plt.legend()
plt.tight_layout()
plt.show()
