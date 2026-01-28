from asyncio.windows_events import NULL
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.optimize import curve_fit
from scipy import odr
from scipy.optimize import minimize
from scipy import stats
from uncertainties import ufloat, ufloat_fromstr
from uncertainties.umath import *  # sin(), etc.
from uncertainties import unumpy

# Table 1 
a_v = ufloat_fromstr("15.40(0.014)")
a_s = ufloat_fromstr("16.71(0.042)")
a_C = ufloat_fromstr("0.701(0.001)")
a_a = ufloat_fromstr("22.56(0.037)")
a_p = ufloat_fromstr("11.88(0.823)")

m_n = 939.565 # [MeV]/[c^2]
c = 299792458

def B(A,Z):
    """
    Semi-Empirical mass formula (2)
    """
    
    delt = (1 - A%2)*(1 - 2*Z%2)    # pairing, add BE for even-even, remove for odd-odd, else 0
    
    vol = a_v * A
    surf = a_s * A**(2/3)
    coul = a_C * Z*(Z-1) * A**(-1/3)
    sym = a_a * (A-2*Z)**2 / A
    pair = a_p * delt * A**(-1/2)

    return vol - surf - coul - sym + pair

print("\Delta energy = ", B(91,36) + B(142,56) - B(235,92), "[MeV]")
print("\Delta m) = ", (B(91,36) + B(142,56) - B(235,92)) / 931.5, "[u]") # m[u] = B[MeV/c^2]/931.5[MeV/c^2u]
print("BE(16O) = ", B(16, 8))

def mass(A,Z):
     
    N = A-Z
    m_n = 939.565 # [MeV]/c^2
    m_H = 938.783 # [MeV]/c^2 
    
    return ((N*m_n + Z*m_H - B(A,Z)) / 931.5)  # translate to [u]   # OBS!!! no (1/c^2) \times B gave right value, does this make sense unit wise?????

#\delta M : n + 235 U --> 91 Kr + 142 Ba + 3n
Delta_M = (mass(235,92) + m_n/931.5) - (mass(91,36) + mass(142, 56) + 3*m_n/931.5)
print("\Delta m = ", Delta_M, "[u]") 


# Energy release per fission:
print("\Delta E = ", Delta_M*931.5, "[MeV]")

# 1 Da [u] = 931.4941 MeV/c^2


# calculation using tabulated values
M_U = 235.0439
M_Kr = 90.9232 
M_Ba = 141.9165
M_n = 1.0087

diff = (M_U + M_n) - (M_Kr + M_Ba + 3*M_n)
print("Tabulated mass diff = ", diff, "[u]")
print("Tabualted energy diff = ", diff*931.5, "[MeV]")



# reactor

P = 0.2*300 * 10**6 # 20% of P [J/S]
E = Delta_M * 931.5 # energy / fission

R = P * (1/(1.60*1e-13)) / E    # fissions/s

R2 = R*3.155*1e7    # [fission events per year]

Myear = R2*0.2350439/(6.022*1e23)   # mass needed for one reactor per year
print("Mass needed in one reactor annually = ",Myear, "kg")

Num_ships = 80

M_navy = Myear*80
volume = M_navy / (1000 * 19.1)

print("Mass needed for entire US navy = ", M_navy, "kg")
print("volume of said U-235 = ", volume, "m^3")


frac = (diff - Delta_M)/Delta_M
print(frac)

print(Delta_M*1.013)
print(diff)



