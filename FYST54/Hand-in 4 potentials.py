import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson 
from scipy.integrate import odeint
import scipy


R0 = 5
V0 = 50

def V_square(r):
    return np.piecewise(r, (r<=R0, r>R0), (-V0, 0))

def V_WS(r):
    a = 0.5
    return -V0 / (1 + np.exp((r - R0)/a))

def normalize(func):
    norm = scipy.integrate.simpson(abs(func)**2)
    return func/np.sqrt(norm)

l = 3
mu = 0.05
k = 1

def Bessel_sq(S,r):

    V = V_square(r)

    chi, z = S

    dchi = z
    dz = chi*(l*(l+1)/r**2 + 2*mu*V - k**2)

    return [dchi, dz]



def Bessel_WS(S,r):
   
    V = V_WS(r)

    chi, z = S

    dchi = z
    dz = chi*(l*(l+1)/r**2 + 2*mu*V - k**2)

    return [dchi, dz]


r0 = 0.001
r = np.linspace(r0, 30, 10000)

chi0 = r[0]*l
z0 = l*r[0]**(l-1)
S0 = [chi0, z0]



#chi_free = 1000*normalize(odeint(Bessel_V0, S0, r).T[0])
chi_sq = 1000*normalize(odeint(Bessel_sq, S0, r).T[0])
chi_WS = 1000*normalize(odeint(Bessel_WS, S0, r).T[0])


#bess = normalize(odeint(Bessel_V0, S0, r).T[0])
#plt.plot(r, bess/r)
#plt.show()


# ind_sc = np.where(chi_WS == max(chi_WS[8000:]))[0]
# print(r[ind_sc])

# ind_fr = np.where(chi_free == max(chi_free[8000:]))[0]
# print(r[ind_fr])

# print(f"\delta_{l} = ", np.round(float(r[ind_fr] - r[ind_sc]),2), "fm")


# plt.plot(r, V_WS(r))
# plt.plot(r, chi_WS)
# plt.plot(r, chi_free, "--")
# plt.xlabel("r [fm]")
# plt.ylabel("V(r) [MeV]/chi(r) [arb. unit]") 
# plt.show()


def oneparam(l):
    def Bessel_V0(S,r):
   
        V = 0

        chi, z = S

        dchi = z
        dz = chi*(l*(l+1)/r**2 + 2*mu*V - k**2)

        return [dchi, dz]
    return Bessel_V0

for l in range(4):
    bessel = oneparam(l)
    
    chi0 = r[0]**(l+1)
    z0 = (l+1)*r[0]**(l)
    
    S0 = [chi0, z0]
    
    chi = odeint(bessel, S0, r).T[0]
    R = normalize(chi/r)

    plt.plot(r, R)
    
plt.show()
