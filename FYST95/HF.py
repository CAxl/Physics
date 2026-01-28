import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


class Nucleus:
	def __init__(self, A, Z):
		self.A = A
		self.Z = Z
		self.N = A - Z
		self.R = 1.1 * pow(self.A, 1/3) # approx of radius, implement Rrms later

	def BE(self):
		delt = (1 - (self.A % 2)) * (1 - (2 * (self.Z % 2)))
	
		a_V = 15.40
		a_s = 16.71
		a_C = 0.701
		a_a = 22.56
		a_p = 11.88
		
		t1 = a_V * self.A
		t2 = a_s * pow(self.A, 2/3)
		t3 = a_C * self.Z * (self.Z - 1) * pow(self.A, -1/3)
		t4 = a_a * pow(self.A - 2*self.Z, 2) / self.A
		t5 = a_p * delt * pow(self.A, - 1.0 / 2.0)
	
		B = t1 - t2 - t3 - t4 + t5
	
		return B

	def m(self):
		mN = 939.565420
		mH = 938.78

		return self.Z * mH + self.N * mN - self.BE()
	
	def nucleonDensity(self):
		rho0 = 0.17
		a = 0.57
		
		r = np.linspace(0, self.R+5)
		den = 1 + np.exp((r-self.R)/a)
		
		
		rho = rho0 / den
		plt.figure(figsize=(8,5))
		plt.plot(r, rho, linestyle='--')
		plt.xlabel("r [fm]")
		ylabel = "rho(r) [nucleons / fm^3]"
		plt.ylabel(ylabel)
		plt.title(f"Nucleon density for A = {self.A}")
		plt.grid(True)
		plt.show()
	
	def BindingEnergyVsA(self, A_max, per_nucleon=False, return_Z=False):
		"""
		Plot BE (or BE/A if per_nucleon=True) vs A from A=1..A_max.
		return_Z: if True, also return the chosen Z for each A as a list.
		"""
		A_vals = np.arange(1, A_max + 1, dtype=int)
		BE_vals = []
		Z_chosen = []
		
		print(f"{'A':>3}  {'best Z':>6}  {'best N':>6}  {'BE (MeV)':>12}  {'BE/A (MeV)':>12}")
		print("-" * 46)

		for A in A_vals:
			# ensure at least one Z is tested:
			# allow Z from 1 .. A-1 normally, but for A==1 use Z=1 (proton) as a sensible fallback

			z_min = 1
			z_max = max(1, A - 1)   # ensures range is non-empty even for A=1

			max_BE = -np.inf
			best_Z = z_min

			for Z in range(z_min, z_max + 1):
				self.A = A
				self.Z = Z
				self.N = A - Z

				be = self.BE()
				if be > max_BE:
					max_BE = be
					best_Z = Z

			# After the Z-loop, max_BE holds the best binding energy for this A
			# store per-nucleon or total BE
			BE_vals.append(max_BE / A if per_nucleon else max_BE)
			Z_chosen.append(best_Z)

			best_N = A - best_Z
			print(f"{A:3d}  {best_Z:6d}  {best_N:6d}  {max_BE:12.3f}  {max_BE/A:12.3f}")
		
		print(round(max(BE_vals),3))
		

		plt.figure(figsize=(8,5))
		plt.plot(A_vals, BE_vals, marker='.', linestyle='-')
		plt.xlabel("Mass Number A")
		ylabel = "Binding Energy per nucleon (MeV)" if per_nucleon else "Binding Energy (MeV)"
		plt.ylabel(ylabel)
		plt.title("Liquid Drop Binding Energy Curve (optimal Z per A)")
		plt.grid(True)
		plt.show()

		if return_Z:
			return A_vals, np.array(BE_vals), np.array(Z_chosen)
		else:
			return A_vals, np.array(BE_vals)

nuc = Nucleus(A=1, Z=1)

# For standard textbook plot, binding energy per nucleon
nuc.BindingEnergyVsA(200, per_nucleon=True)

NN = Nucleus(100, 50)
NN.nucleonDensity()


class HartreeFock(Nucleus):
	def __init__(self, A, Z, rho0):
		super().__init__(A, Z)
		self.rho0 = rho0

	def printer(self):
		print("Initialized nucleus with:")
		print("A = ", self.A)
		print("Z = ", self.Z)
		print("N = ", self.N)


N1 = HartreeFock(19, 9, 1)
N1.printer()
