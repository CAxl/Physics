import numpy as np



hbarc = 197.326
mc2 = 938.272
A = 114 # Sn

oneoverbsquared = mc2 / (hbarc**2) * 41 * pow(A,-1/3)
b = np.sqrt(1/oneoverbsquared)

print("b = ", b, " fm")

r_eff = b*np.sqrt(13/2)

print("r_eff = ", r_eff, " fm")

R = 1.2*pow(A,1/3)
print("R = ", R, " fm")


r_eff_2d = b * np.sqrt(11/2)
print(r_eff_2d)


def U(r, Z):
    const = hbarc / 137
    R = pow(A, 1/3)
    
    return const * (Z/R) * ((-1/2)*(pow(r/R,2.0)) + 3/2)

print(U(r_eff, 50), "MeV")
print(U(r_eff_2d, 50), "MeV")





# question one

def X(A):
    a_c = 0.705
    a_s = 17.944
    a_v = 15.494
    k_v = 1.7826

    num = a_c * pow(A, 5/3)
    den = 4*a_v*k_v - 4*a_s*k_v*pow(A,-1/3) + a_c*pow(A,2/3)
    

    X = num / den # X = N-Z
    return X

print("Numerical values exercise 1")
print("A = 100: ", X(100))
print("A = 200: ", X(200))
print("A = 300: ", X(300))




