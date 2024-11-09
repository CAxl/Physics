import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

# Nitrogen exchange energies of the two excitations
# N_3S = 15.427067
# N_1S = 15.670446  

# N_3P = 15.666684 
# N_1P = 15.827774 

# K_NS = (N_1S - N_3S) / 2
# K_NP = (N_1P - N_3P) / 2

# # Oxygen exchange energies of the two excitations
# O_3S = 20.619028
# O_1S = 20.906678 

# O_3P = 20.898503
# O_1P = 21.092680 

# K_OS = (O_1S - O_3S) / 2
# K_OP = (O_1P - O_3P) / 2

# print("Exchange energy Nitrogen for 1s2s = ", K_NS)
# print("Exchange energy Nitrogen for 1s2p = ", K_NP)
# print("Exchange energy Oxygen for 1s2s = ", K_OS)
# print("Exchange energy Oxygen for 1s2p = ", K_OP)


# # Constants. 
m_n = 1.6749 * 10**(-27) 
m_p = 1.6726 * 10**(-27) 
m_e = 9.1093837 * 10**(-31) 
e = 1.60217663 * 10**(-19) 
hbar = 1.05457182 * 10**(-34) 
eps0 = 8.854187817 * 10**(-12)
 



# # Oxygen
# Z = 8
# m_N = Z*m_p + Z*m_n 
# mu = m_e*m_N/(m_e + m_N)
# R_mu = get_R_mu(mu)

# def E_n(n):
#     return - (Z**2/n**2)*R_mu

# def E_nj(n, j):
#     En = E_n(n)
#     return En - (a**2 * Z**2 /n**2)*En*((3/4) - n/(j + 1/2))

# K1s2s = ((568.899694 - 561.072349)/2)*e
# deltaE = 561.072349*e

# E01s = E_nj(1, 1/2)
# E01ss = 2*E01s

# E02s = E_nj(2, 1/2)
# E01s2s = E01s + E02s 

# deltaJ = E01s2s - deltaE - K1s2s - E01ss 
# print(deltaJ/e) 


# # Nitrogen
# Z = 7
# m_N = Z*m_p + Z*m_n 
# mu = m_e*m_N/(m_e + m_N)

# def E_n(n):
#     return - (Z**2/n**2)*R_mu

# def E_nj(n, j):
#     En = E_n(n)
#     return En - (a**2 * Z**2 /n**2)*En*((3/4) - n/(j + 1/2))

# K1s2s = ((426.414570 - 419.791888)/2)*e
# deltaE = 419.791888*e

# E01s = E_nj(1, 1/2)
# E01ss = 2*E01s

# E02s = E_nj(2, 1/2)
# E01s2s = E01s + E02s 

# deltaJ = E01s2s - deltaE - K1s2s - E01ss 
# print(deltaJ/e) 


# d


J = 1
g_s = 2




# energies for B-field
CP1 = 304.4034097*e # carbon
CP0 = 304.4049606*e
NP1 = 426.2922780 *e  # nitrogen
NP0 = 426.2912023 *e 
O1s2p_3P_1 = 568.640052 *e # oxygen
O1s2p_3P_0 = 568.632762 *e
FP1 = 731.470216 * e    # Flourine
FP0 = 731.451468 * e 
Ne1 = 914.802870 * e # neon
Ne0 = 914.765595 * e 
Na1 = 1118.671115 * e # sodium
Na0 = 1118.606328 * e
Mg1 = 1343.098984 * e #magnesium Z = 12
Mg0 = 1342.995806 * e
Al1 = 1588.12560 * e    # aluminium
Al0 = 1587.97098 * e





def B(level1, level2):
    return (2*np.sqrt(2)*m_e*(np.abs(level1 - level2))) / (e*hbar*J)    # abs for energy difference


print(B(O1s2p_3P_1,O1s2p_3P_0))
print(B(NP1, NP0))
print(B(CP1,CP0))
print(B(Na1,Na0))


# plotting K: 1s2s K = 1S - 3S [eV!!!!!!!!!]
CK = (1/2)*(304.387768 - 298.962205)
NK = (1/2)*(426.414570 - 419.791888)
OK = (1/2)*(568.899694 - 561.072349)
FK = (1/2)*(731.86274 - 722.82326)
NeK = (1/2)*(915.32141 - 905.06210)
NaK = (1/2)*(1119.30514 - 1107.81772)
MgK = (1/2)*(1343.83638 - 1331.11204)
AlK = (1/2)*(1588.95035 - 1574.97995)

# plotting J:
CdE = (298.962205)
NdE = (419.791888)
OdE = (561.072349)
FdE = 722.82326
NedE = 905.06210
NadE = 1107.81772
MgdE = 1331.11204
AldE = 1574.97995

def get_R_mu(mu):    # Rydberg reduced mass
    return (1/2)*mu*e**4/((4*np.pi*eps0)**2 *hbar**2) 

def E_0(Z):
    a = 1/137
    #mu = (m_e*(m_p+m_e))/(m_e*(m_p+m_n))
    #R_mu = get_R_mu(mu)
    R_mu = 1.09678 * 10**7
    E1s = ((-(Z**2)*R_mu) + ((a**2)*R_mu*(Z**4)*((3/4)-1)))*e
    E2s = ((-(Z**2)*(1/4)*R_mu) + ((1/16)*(a**2)*R_mu*(Z**4)*((3/4)-2)))*e
    E1s2s = E1s + E2s 
    E1s2 = 2*E1s 
    return E1s2 - E1s2s

E0C = E_0(6)
E0N = E_0(7)
E0O = E_0(8)
E0F = E_0(9)
E0Ne = E_0(10)
E0Na = E_0(11)
E0Mg = E_0(12)
E0Al = E_0(13)

print(E0N + NK + NdE)

CJ = E0C + CK + CdE
NJ = E0N + NK + NdE
OJ = E0O + OK + OdE
FJ = E0F + FK + FdE
NeJ = E0Ne + NeK + NedE
NaJ = E0Na + NaK + NadE
MgJ = E0Mg +MgK + MgdE
AlJ = E0Al + AlK + AldE

# e
# [C, N, O, F, Ne, Na, Mg, Al]
Klist = [CK, NK, OK, FK, NeK, NaK, MgK, AlK]
Jlist = [CJ, NJ, OJ, FJ, NeJ, NaJ, MgJ, AlJ]
Blist = [B(CP1,CP0), B(NP1,NP0), B(O1s2p_3P_1,O1s2p_3P_0), B(FP1,FP0), B(Ne1,Ne0), B(Na1,Na0), B(Mg1,Mg0), B(Al1,Al0)]
Zlist = [6,7,8,9,10,11,12,13]

plt.scatter(Zlist,Blist)
fit = np.poly1d(np.polyfit(Zlist,Blist,4))
fitline = np.linspace(6,13,100)
plt.plot(fitline, fit(fitline), 'orange', linestyle='--',label=r"$\sim \alpha Z^4$")
plt.xlabel(r"$Z$")
plt.ylabel(r"$B(Z)$ [T]")
plt.legend()
plt.show()

plt.scatter(Zlist, Klist)
plt.xlabel(r"Z")
plt.ylabel(r"K(Z) [eV]")
fit1 = np.poly1d(np.polyfit(Zlist,Klist,1))
line = np.linspace(6,13,100)
plt.plot(line,fit1(line), 'orange', linestyle='--',label=r"$\sim kZ$")
plt.legend()
plt.show()

plt.scatter(Zlist, Jlist)
plt.xlabel(r"Z")
plt.ylabel(r"J(Z) [eV]")
model = np.poly1d(np.polyfit(Zlist,Jlist,2))
polyline = np.linspace(6, 13, 100)
plt.plot(polyline,model(polyline),'orange',linestyle='--',label=r"$\sim k Z^2$")
plt.legend()
plt.show()



