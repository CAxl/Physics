import numpy as np
import matplotlib.pyplot as plt


def B(Z,A):
	N = A - Z
	delt = (1 - (A % 2)) * (1 - (2 * (Z % 2)))
	
	a_V = 15.40
	a_s = 16.71
	a_C = 0.701
	a_a = 22.56
	a_p = 11.88
		
	t1 = a_V * A
	t2 = a_s * pow(A, 2/3)
	t3 = a_C * Z * (Z - 1) * pow(A, -1/3)
	t4 = a_a * pow(A - 2*Z, 2) / A
	t5 = a_p * delt * pow(A, - 1.0 / 2.0)
	

	B = t1 - t2 - t3 - t4 + t5
	
	return B


print(B(8,16))

def m(Z,A):
	N = A - Z
	
	mN = 939.565420
	mH = 938.78

	return Z * mH + N * mN - B(Z, A)


def rho(r, A):
	
	rho0 = 0.17
	R = 1.1 * pow(A, 1/3)
	a = 0.57

	rho = rho0 * 1/(1 + np.exp((r-R)/a))
	
	return rho


import numpy as np

 
def rho(r, A):
    rho0 = 0.17
    R = 1.1 * pow(A, 1/3)
    a = 0.57
    return rho0 / (1 + np.exp((r - R) / a))



def R_half_density(A, r):
    """
    Returns the radius R where rho(r) = 0.5 * rho0 (half-density radius).
    Uses linear interpolation between grid points.
    """
    rho_vals = rho(r, A)
    rho0 = 0.17
    target = 0.5 * rho0

    # Find index where rho crosses 0.5 * rho0
    idx = np.where(rho_vals <= target)[0][0]  # first point where rho <= target

    # Linear interpolation between r[idx-1] and r[idx]
    r1, r2 = r[idx-1], r[idx]
    rho1, rho2 = rho_vals[idx-1], rho_vals[idx]

    # Interpolate
    R_half = r1 + (target - rho1) * (r2 - r1) / (rho2 - rho1)
    return R_half



def Rrms(A, r):
    """
    Compute the RMS radius:
        R_rms = sqrt( (5/3) * <r^2> )
    where <r^2> = ∫ rho(r) r^2 dr / ∫ rho(r) dr

    Parameters:
        A (int): mass number
        r (np.array): radial grid (must be monotonically increasing, in fm)

    Returns:
        float: R_rms value in fm
    """
    rho_vals = rho(r, A)

    # Numerators and denominators for <r^2>
    num = np.trapz(rho_vals * r**2, r)
    den = np.trapz(rho_vals, r)

    r2_mean = num / den

    R_rms = np.sqrt((5.0/3.0) * r2_mean)
    return R_rms



rspace = np.linspace(0, 10, 2000)
A = 100
rho0 = 0.17
a = 0.57
R_rms = Rrms(A, rspace)
R_half_val = R_half_density(A, rspace)
R = 1.1 * pow(A, 1/3)
pref = np.pi**2 * (7/6)
R_expansion = R * (1 + pref*(a/R)**2) 
print("root mean square radius = ", R_rms, " fm")
print("Half density radius = ", R_half_val, " fm")
print("R = 1.1A^1/3 = ", R, " fm")
print("R = 1.2A^1/3 = ", 1.2 * pow(A, 1/3), " fm")
print("Rrms = R[1 + ...] = ", R_expansion, " fm")

plt.plot(rspace, rho(rspace, 16), label = "A = 16")
plt.plot(rspace, rho(rspace, 32), label = "A = 32")
plt.plot(rspace, rho(rspace, 100), label = "A = 100")
# Dashed line starting from y-axis (x=0) to intersection (x=R_half)
plt.hlines(0.5 * rho0, xmin=0, xmax=R_half_val, linestyles="--", color = "black", label=r"$\rho = 0.5\rho_0$")
plt.vlines(R_half_val, ymin=0, ymax=0.5 * rho0, linestyles="--", color = "black")

plt.ylabel("rho(r) [nucleons/fm^3]")
plt.xlabel("r [fm]")
plt.title("Fermi function for nucleon density")

plt.legend()
plt.show()


