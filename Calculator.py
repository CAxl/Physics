import numpy as np
# import matplotlib.pyplot as plt 
# from scipy.optimize import minimize
# from scipy import stats
# from uncertainties import ufloat, ufloat_fromstr
# from uncertainties.umath import *  # sin(), etc.
# from uncertainties import unumpy
# import numpy as np
# from pylab import *
# from scipy.optimize import curve_fit
# from scipy import odr
# import sys
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# from scipy import misc
# import scipy as scp
# import sympy as sym

# # table of constants 
# hbar = 6.62607015 * 1e-34 / (2*np.pi)
# c = 299792458
# k_B = 1.380649 * 1e-23
# epsilon_0 = ufloat_fromstr("8.8541878128(13)") * 1e-12
# m_e = 9.1093837 * 1e-31
# m_p = 1.67262192 * 1e-27
# m_n = ufloat_fromstr("1.674927351(74)") * 1e-27
# e = 1.60217663 * 1e-19
# alpha = 1/137
# u = 1.66053907*1e-27



# m_trit = 3.01604928*u
# m_He = 3.016*u
# m_p = 1.00784 * u
# m_Bor = 10.0129370*u  
# m_Li = 7.01601*u  
# m_alpha = 4.002603*u


# # a + A --> b + B
# Q1 = (m_n + m_He - m_trit - m_p)*c**2 /(1e6*e)
# Q2 = (m_n + m_Bor - m_Li - m_alpha)*c**2 /(1e6*e)
# Q3 = Q2 - 0.477 # excitation energy of Li^*

# print("Q1 = ", Q1, " MeV")
# print("Q2 = ", Q2, " MeV")
# print("Q3 = ", Q3, " MeV")
# print("_________________")

# def get_Ekin(mb,mB,Q):
#     return mB*Q/(mb+mB)

# KE_trit = get_Ekin(m_trit,m_p,Q1)
# KE_proton = get_Ekin(m_p,m_trit,Q1)
# KE_lit = get_Ekin(m_Li,m_alpha,Q2)
# KE_alpha2 = get_Ekin(m_alpha,m_Li,Q2)
# KE_lit_star = get_Ekin(m_Li,m_alpha,Q3)
# KE_alpha3 = get_Ekin(m_alpha,m_Li,Q3)

# print("E_kin(trit) = ", KE_trit, "MeV")
# print("E_kin(proton) = ", KE_proton, "MeV")
# print("E_kin(lithium) = ", KE_lit, "MeV")
# print("E_kin(alpha2) = ", KE_alpha2, "MeV")
# print("E_kin(lithium*) = ", KE_lit_star, "MeV")
# print("E_kin(alpha3) = ", KE_alpha3, "MeV")

# # rad env

# z = 2   # atomic number of \alpha particle
# m_ec2 = 0.511 # MeV
# Z = 14.46   # atomic number of the material
# rho = 1.225 * 1e-3  # density of material (air) g / cm^3
# A = 28.96  # g/mol
# I = 8.6 * 1e-5  # MeV
# m_alphc2 = 3727 # MeV
# G = 0.307075 # (e**2/(4*np.pi*epsilon_0))**2 * (4pi N_A / m_e c^2) [MeV cm^2 / g]

# E = np.linspace(1,10,1000)
# v = 1
# beta = v**2 / c**2

# dE = G * z**2 * Z * rho /(beta**2 * A) * (np.log(2*m_e * c**2 * beta**2 / I) - np.log(1-beta**2) - beta**2)




# plt.plot(E,dE)
# plt.show()

# deltE1 = 40 * 1e3
# deltE2 = 150 * 1e3
# V1 = 30*1e3
# V2 = 100 * 1e3

# def Rval(E,V):
#     return E/V

# delta1 = deltE1*np.sqrt(1 + (4/Rval(deltE1, V1)**2))
# delta2 = deltE1*np.sqrt(1 + (4/Rval(deltE1, V2)**2))
# delta3 = deltE2*np.sqrt(1 + (4/Rval(deltE2, V1)**2))
# delta4 = deltE2*np.sqrt(1 + (4/Rval(deltE2, V2)**2))

# print(delta1/1e3)   # print in [keV]
# print(delta2/1e3)
# print(delta3/1e3)
# print(delta4/1e3)


# ########### 11 Neutron flux ############

# eps = 3.0 * 1e-4    # efficiency of detector
# counts = 5.00 * 1e4 
# T = 15 * 60 + 3600  # 15 min + 1 h [s]
# t_half = 54 * 60    # half life 116^In  [s]

# # produced radioactive nuclei
# N_prod = counts / (eps*(np.exp(-np.log(2)*15*60/t_half) - np.exp(-np.log(2)*T/t_half)))
# print("tsjhefj",N_prod)

# # production rate
# P = N_prod * np.log(2) / (t_half*(1-np.exp(-np.log(2)*60/t_half)))
# print(P)

# rho = 7.310 * 1000   # density of In [kg/m^3]
# M_115In = 114903878.773 * 1e-6 * 1.66053907 * 1e-27     # M(^115In) [kg]
# V = 3*3*1e-6 * 1e-6    # volume of 3x3 mm^2 x 1µm foil [m^3]

# N0 = rho*V/M_115In  # number of target nuclei 115^In
# print("hej", N0)

# I = P/(N0 * 160 * 1e-28) /(100*100)    # flux in [cm^-2 s^-1]
# print("flux I = ", I / 1e11, "* 10^11 cm^-2s^-1")



# ########### 12 Coulomb scattering ###############

# T = 9   # MeV
# theta1 = np.pi/4 
# theta2 = np.pi/2 
# theta3 = 3*np.pi/4
# m_p = 938.2720813   # MeV/c^2

# def p_f(theta):
#     return np.sqrt(2*m_p*T) - (4*T*np.sin(theta/2) / (np.sqrt(2*T/m_p)))

# E_f1 = p_f(theta1)**2/(2*m_p)
# E_f2 = p_f(theta2)**2/(2*m_p)
# E_f3 = p_f(theta3)**2/(2*m_p)
# print("E(45) = ", E_f1, "MeV")
# print("E(90) = ", E_f2, "MeV")
# print("E(135) = ", E_f3, "MeV")


# ### b ###

# T_a = 9  # MeV
# M_7li = 7.016003434 * 931.49410242  # M(7Li) [MeV/c^2]
# theta1 = np.pi/4 
# theta2 = np.pi/2 
# theta3 = 3*np.pi/4
# m_p = 938.2720813   # MeV/c^2

# def r(theta):
#     return np.cos(theta)*np.sqrt(m_p*m_p*T_a) / (m_p + M_7li)

# s = T_a*(M_7li - m_p)/(m_p + M_7li)

# T_b1 = r(theta1) + np.sqrt(r(theta1)**2 + s)    # sqrt of E_kin of the ejectiles at different angles
# T_b2 = r(theta2) + np.sqrt(r(theta2)**2 + s)
# T_b3 = r(theta3) + np.sqrt(r(theta3)**2 + s)
 
# print(T_b1**2)
# print(T_b2**2)
# print(T_b3**2)

# Q = -0.477  # endothermic reaction

# s = (Q*M_7li + T_a*(M_7li - m_p))/(m_p + M_7li)
# T_p = r(theta2) + np.sqrt(r(theta2)**2 + s)

# print(T_b2**2 + Q)

# print(T_p**2)


# ### c ###

# e = 1.60217663 * 1e-19
# A = 197
# R_Au = 1.2 * A**(1/3)   # fm
# R_p = 0.84  # fm
# print(R_Au)

# Rtot = (R_p + R_Au)*1e-15


# V_C = 79*e**2 / (4*np.pi*epsilon_0*Rtot)
# print(V_C / (e*1e6))

# ### d ###

# R_Ca = 1.2 * 48**(1/3) # fm
# R_Am = 1.2 * 243**(1/3) # fm

# Rtot = (R_Ca + R_Am)*1e-15  # m

# V_C = 20*95*e**2 / (4*np.pi*epsilon_0*Rtot)

# print(V_C / (e*1e6))


# ### e ###

# u = 1.66053907*1e-27
# e = 1.60217663 * 1e-19

# M_X = 243061379.9 * 1e-6 * u
# M_a = 47952522.654 * 1e-6 * u
# M_Y = (290196240 * 1e-6 + 1.0087) * u  # M(Mc-290) + M(n)

# Q = (M_X + M_a - M_Y)*c**2 / (1e6*e)    # Q-value in MeV

# print("Q = ", Q, "MeV")

# E_ex = Q + 245 
# print(E_ex)



# rad env

# avg = np.array([(0.9*0.12 + 0.9*0.04 + 0.8*0.2)/(2.6), (0.9*0.01 + 0.9*0.03 + 0.8*0.10)/(2.6), (0.9*0.07 + 0.9*0.1 + 0.8*0.16)/(2.6), (0.9*0.01 + 0.9*0.01 + 0.8*0.13)/2.6, (0.9*0.01 + 0.9*0.02 + 0.8*0.15)/2.6, (0.9*0.14 + 0.9*0.12 + 0.8*0.12)/2.6])

# PED+ has 10% uncertainty
# RNI 10/R 20% uncertainty
#weighted_err = np.array([(0.9*0.12 * 0.1 + 0.9*0.04 * 0.1 + 0.8*0.2*0.2)/(2.6) , (0.9*0.01 * 0.1 + 0.9*0.03 * 0.1 + 0.8*0.10 * 0.2)/(2.6) , (0.9*0.07*0.1 + 0.9*0.1*0.1 + 0.8*0.16*0.2)/(2.6) , (0.9*0.01 + 0.9*0.01 + 0.8*0.13)/2.6, (0.9*0.01*0.1 + 0.9*0.02*0.1 + 0.8*0.15*0.2)/2.6, (0.9*0.14*0.1 + 0.9*0.12*0.1 + 0.8*0.12*0.2)/2.6])

# PED+ device 1
# p11 = ufloat(0.12, 0.1*0.12)
# p12 = ufloat(0.01, 0.1*0.01)
# p13 = ufloat(0.07, 0.1*0.07)
# p14 = ufloat(0.01, 0.1*0.01)
# p15 = ufloat(0.01, 0.1*0.01)
# p16 = ufloat(0.14, 0.1*0.14)

# PED+ device 2
# p21 = ufloat(0.04, 0.1*0.04)
# p22 = ufloat(0.03, 0.1*0.03)
# p23 = ufloat(0.1, 0.1*0.1)
# p24 = ufloat(0.01, 0.1*0.01)
# p25 = ufloat(0.02, 0.1*0.02)
# p26 = ufloat(0.12, 0.1*0.12)

# RNI device
# R1 = ufloat(0.2, 0.2*0.2)
# R2 = ufloat(0.1, 0.2*0.1)
# R3 = ufloat(0.16, 0.2*0.16)
# R4 = ufloat(0.13, 0.2*0.13)
# R5 = ufloat(0.15, 0.2*0.15)
# R6 = ufloat(0.12, 0.2*0.12)


# weighted_avg = np.array([(0.9*p11 + 0.9*p21 + 0.8*R1)/(2.6), (0.9*p12 + 0.9*p22 + 0.8*R2)/(2.6), (0.9*p13 + 0.9*p23 + 0.8*R3)/(2.6), (0.9*p14 + 0.9*p24 + 0.8*R4)/2.6, (0.9*p15 + 0.9*p25 + 0.8*R5)/2.6, (0.9*p16 + 0.9*p26 + 0.8*R6)/2.6])
# annual_effective_dose_rate = weighted_avg*pow(10,-3)*24*365 
# print(annual_effective_dose_rate)


# annual_effective_dose_rate_indoors = annual_effective_dose_rate[:3]
# coef = 6.7*pow(10,-6)*365*24 # mSv per Bqhm^3 
# print(annual_effective_dose_rate_indoors*coef)
# print(annual_effective_dose_rate_indoors*coef/annual_effective_dose_rate_indoors)


N = 1000
L = 100

r = np.linspace(0.01, L, N)
dr = np.abs(r[1]-r[0])

print(r)

print("dr = ", dr)




# def V(r):
#     """
#     Harmonic oscillator potential
#     """
#     E0 = -55    # MeV    
#     hw = 8.6
#     mc2 = 939.57
#     hc = 197.326

#     const = 2*mc2 / (hc**2) # const infront of V(r) in Radial SE

#     V = ( 0.5 * (mc2)*(hw**2)* r**2 / (hc**2) ) + E0

#     return const * V

# diag = np.diag(V(r))
# print(diag)


