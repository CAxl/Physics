import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
from scipy import stats
from uncertainties import ufloat, ufloat_fromstr
from uncertainties.umath import *  # sin(), etc.
from uncertainties import unumpy
import numpy as np
from pylab import *
from scipy.optimize import curve_fit
from scipy import odr
import sys
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy import misc
import scipy as scp


# table of constants 
hbar = 6.62607015 * 1e-34 / (2*np.pi)
c = 299792458
k_B = 1.380649 * 1e-23
epsilon_0 = ufloat_fromstr("8.8541878128(13)") * 1e-12
m_e = 9.1093837 * 1e-31
m_p = 1.67262192 * 1e-27
m_n = ufloat_fromstr("1.674927351(74)") * 1e-27
e = 1.60217663 * 1e-19
alpha = 1/137
u = 1.67377 * 1e-27


############### 5a ########################

A = 2 * 1e-6 * 3.7 * 1e10   # activity in Bq
T = 8 * 60 * 60             # time in s
E_gamma = 0.5 * e * 1e6     # energy in J
mu_over_rho = 8.712*1e-3    #cm2/g
rho_air = 1.293 * 1e-3      #g/cm3
mu = rho_air*mu_over_rho * 100  # 1/m
m = 75  # kg

N0 = A*T # number of photons emitted over 8 hrs

def D(x):
    return 1.7*0.5*(1/(4*np.pi*x**2)) * E_gamma*N0*np.exp(-mu*x) / m

# print("dose(x=1) = ", D(1), "J/kg")
# print("dose(x=4) = ", D(4), "J/kg")

################## 5b ########################

m = 75  # mass of person [kg]

T_14C = 5700        # half life for carbon-14 [years]
abund_14C = 1e-12   # nat. abundance of carbon-14
m_14C = 14003241.989 * 1e-6 * 1.66053907 * 1e-27 # m(^14C) [kg]

N0_14C = 0.2*m*abund_14C / m_14C 

T_40K = 1.248 * 1e9     # half life for potassium-40 [years]
abund_40K = 0.000117    # nat. abundance f
m_40K = 39963998.17 * 1e-6 * 1.66053907 * 1e-27 # m(^40K) [kg]

N_01_40K = (0.0025 * 0.89 * abund_40K * m) / m_40K
N_02_40K = (0.0025 * 0.11 * abund_40K * m) / m_40K

# def get_avgA(N0,T): 
#     """ takes initial number of nuclei N0 and half-life T [y]
#     --> returns average activity over 80 yrs"""
#     Avals = []
#     lifetime = np.arange(1,81,1)
    
#     for t in lifetime:
#         Avals = np.append(Avals, N0 * np.exp(-(t/T)*np.log(2)) * np.log(2) / (T * 31556926))
    
#     return np.average(np.array(Avals))

def get_A(N0,T):
    return N0 * np.log(2) / (T * 31556926)

A_14C = get_A(N0_14C, T_14C)
A1_40K = get_A(N_01_40K, T_40K)
A2_40K = get_A(N_02_40K, T_40K)
print("Activity for Carbon decay = ",A_14C , "Bq") 
print("Activity for decay into ^40Ca = ",A1_40K, "Bq")
print("Activity for decay into ^40Ar = ",A2_40K, "Bq")
print()
print("Total activity = ", A_14C + A1_40K + A2_40K, "Bq")

print("_______________________")


########### 5c ################

# Q_eps = (m_40K - m_40Ar) *c**2 - (1.461 * 1e6 * e)  # last term = E_x
# print("Q_EC = ", Q_eps / (1e6*e), "MeV")

# Q_tot_Ar = (Q_eps / (1e6 * e) + 1.461) * 1e6 * e    # Q-value for K decay into Ar

c = 299792458
e = 1.60217663 * 1e-19
m_40K = 39963998.17 * 1e-6 * 1.66053907 * 1e-27     # m(^40K) [kg]
m_40Ar = 39962383.1220 * 1e-6 * 1.66053907 * 1e-27  # m(^40Ar) [kg]
Tlife = 80 * 31556926   # human lifetime in seconds

Q_C = 0.16 * 1e6 * e    # Q value carbon decay [J]
Q_Ca = 1.3 * 1e6 * e    # Q value K decay into Ca [J]
Q_Ar = (m_40K - m_40Ar) * c**2  # Q-value EC into Ar + E_x(gamma)

D_C = A_14C * Tlife * Q_C / 75  # D = E_tot / m
D_Ca = Q_Ca * Tlife * A1_40K / 75
D_Ar = Q_Ar * Tlife * A2_40K / 75

print("D_carbon = ", D_C, "J/kg")
print("D_potassium = ", D_Ar + D_Ca, "J/kg")
print()
print("total dose = ", (D_Ca + D_C + D_Ar) * 1e3 / 80, "mGy / y")


######################### 7b ###########################
# t = 30.1 * 31556926 # years to seconds
# Q = 1.176 # Q-value [MeV]
# E_x = 0.662 # Excitation energy [MeV]

# E_0ex = (Q - E_x )  # end-point energy for ex. state
# print("Endpoint energy to excited state = ", E_0ex, "MeV")
# logf_ex = 0.5  # log(f) corresponding to end-point energy ex. state (Krane fig. 9.8)
# logT_ex = np.log10(t/0.94) # partial half-life for decay into excited state

# logft_ex = logf_ex + logT_ex
# print("log_10(ft) for excited state = ", logft_ex)

# E_0gr = Q   # end-point energy for ground state
# print("Endpoint energy to ground state = ", E_0gr, "MeV")
# logf_gr = 1.5   # log(f) corresponding to end-point energy gr. state (Krane fig. 9.8)
# logT_gr = np.log10(t/0.056) # partial half-life for decay into gr. state
# print("test = ", logT_gr)

# print("log(ft) for gr. state = ", logf_gr + logT_gr)

################ 7c #########################

# m_137Cs = 136907089.3 * 1e-6 * 1.66053907 * 1e-27 
# m_137Ba = 136905827.21  * 1e-6 * 1.66053907 * 1e-27
# m_e = m_e # MeV/c^2
# Q_beta_minus = (m_137Cs - m_137Ba)*c**2
# Q_beta_plus = (m_137Cs - m_137Ba - 2*m_e)*c**2
# print("Q_beta_minus = ", Q_beta_minus, "MeV") 
# print("Q_beta_plus = ", Q_beta_plus, "MeV")


##################### 8 - deuteron ####################

g = 938.9   # 2mu*c^2
E = 2.2
V0 = 35
R = 2.1 
hc = 197.3

k1 = np.sqrt(g*(-E+V0))/hc
k2 = np.sqrt(g*E)/hc

B=1 # arbitrary
C = B*np.sin(k1*R)/np.exp(-k2*R) # defined in terms of B

def u1(r):
    return B*np.sin(k1*r)

def u2(r):
    return C*np.exp(-k2*r)

def u1_mod_sq(r):   # modulus squared of u_1(r)
    return np.abs(B*np.sin(k1*r))**2

def u2_mod_sq(r):   # modulus squared of u_2(r)
    return np.abs(C*np.exp(-k2*r))**2

I1_num = scp.integrate.quad(u1_mod_sq,0,2.1)    # I1
I2_num = scp.integrate.quad(u2_mod_sq,2.1,np.inf)  # I2
print(I2_num[0] / I1_num[0])

I2 = 4*np.pi* (np.exp(-2*k2*R) /(2*k2))*C**2    # analytical solutions
I1 = (R/2 - (np.sin(2*k1*R) / (4*k1)))*4*np.pi*B**2
print(I2 / I1)

x1 = np.linspace(0,2.1,100) # r-range first region
x2 = np.linspace(2.1, 10, 100)  # r-range second region

plt.plot(x1, u1(x1), label="u_1(r)")
plt.plot(x2, u2(x2), label="u_2(r)")
plt.vlines(2.1,0,1, color='black', linestyles="--", label=f"r = {2.1} fm")
plt.ylabel("u(r)")
plt.xlabel("r [fm]")
plt.legend()
plt.show()




# print(k1*(1/np.tan(k1*R)))
# print(-k2)










