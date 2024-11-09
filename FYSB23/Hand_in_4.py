import numpy as np

h = 6.626*10**(-34)
c = 300000000 
hbar = h/(2*np.pi)
m_e = 9.1093837 * 10**(-31)
m_p = 1.67262192 * 10**(-27)
G = 6.6743 * 10**(-11)
M_star = 1.2 * 2 * 10**(30)
k_B = 1.380649 * 10**(-23)


K = (((h/(2*np.pi))**2) * (3*(np.pi**2))**(2/3)) * (1/((2*m_p)**(5/3))) / (5*m_e)
c1 = ((8*np.pi * h**2)/(15*m_e)) * ((3/(8*2*np.pi*m_p))**(5/3))



R_star = (3/(4*np.pi))**(2/3) * ((2*K)/G) * M_star**(-1/3)

print(R_star/1000)

rho = 3*M_star/(4*np.pi*R_star**3)
P = K * rho**(5/3)

E = ((hbar**2)/(2*m_e)) * (((3*rho*np.pi**2)/(2*m_p))**(2/3))
print("T = ", E/k_B/1e10)

Ef = (hbar**2 /(2*m_e))*(((9*M_star*np.pi**2)/(4*np.pi*2*m_p*R_star**3))**(2/3))

gamma = (np.sqrt(1 + ((2*Ef) / (m_e*c**2))))
print(gamma)





