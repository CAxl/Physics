import numpy as np
from matplotlib import pyplot as plt
import analytical_solution as AS
import radial 
import radial_log
import sympy 
import Modified_radiallog_leon as Mrl 
import scipy.optimize as opt


################################################################################################################################################ PART 1 


############## QUESTION 2 ################
"""
l = 0 1 2 3 4 5 6 7 8 . . .
    s p d f g h i k l . . .

- What is meant by the radial wave function P(r)?
P(r) is the radial function which is a factor in the common eigenfunction of the 
Hamiltonian,l^2 and l_z. 

- What is the radial density function D(r) and how do you interpret it?
The radial density function is (P_nl(r))^2. It gives the "probability per unit length"
of finding the electron at a distance r from the nucleus. So we expect that
the integral of the radial density function between a and b gives the probability
of finding the electron in the range (r = a, r = b). 

- Plot the radial density function for 3s, 3p and 3d with Z = 1. Comment on if the 
number of nodes is what is expected. 
For P_3s we have 3 = 0 + v + 1 --> v = 2 i.e we expect two nodes : TRUE
For P_3p we have 3 = 1 + v + 1 --> v = 1 i.e we expect one node: TRUE
For P_3d we have 3 = 2 + v + 1 --> v = 0 i.e we expect 0 nodes: TRUE
"""

# r = np.linspace(0, 30, 1000)
# Z = 1
# P_3s = AS.P3s(r, Z)**2
# P_3p = AS.P3p(r, Z)**2
# P_3d = AS.P3d(r, Z)**2
# plt.plot(r, P_3s, label = "3s")
# plt.plot(r, P_3p, label = "3p")
# plt.plot(r, P_3d, label = "3d")
# plt.title("Radial density functions for 3s, 3p and 3d with Z = 1")
# plt.legend()
# plt.xlabel("r [a.u]")
# plt.ylabel("Probability density")
# plt.show() 

"""
- Plot the effective potential V(r) for s, p, and d electrons. 
"""

# r = np.linspace(0.0001, 5, 1000)
# Z = 1

# def V(r, l):
#     return -Z/r + l*(l+1)/(2*r**2)

# s = V(r, 0) 
# p = V(r, 1)
# d = V(r, 2)
# plt.plot(r, s, label = "s")
# plt.plot(r, p, label = "p")
# plt.plot(r, d, label = "d")
# plt.legend()
# plt.title("Effective potential for s, p and d electrons with Z = 1")
# plt.xlabel("r [a.u]")
# plt.ylabel("V(r) [a.u]")
# plt.ylim(-10, 20)
# plt.xlim(0,5)
# plt.grid()
# plt.show()



"""
- How does the potential and the radial density function compare?
We see a shift in x depending on the strength of V(r) close to r = 0. 

- What makes the potential of the s orbital special and what result does this have on the density close to the nucleus? 
For s, we have l = 0 thus the potential is the pure coloumb potential which is proportional to - 1/r. 
This is reflected in the radial density function: the 3s orbital is the only one which has significant values of the
density function close to r = 0. 

"""


############## QUESTION 3 ################
"""
- Comment on what the difference is between radial.py and radiallog.py. 
radial.py uses a uniform grid r_k = r_0 + kh where h is (constant) stepsize. 
To increase the numerical accuracy, we want more gridpoints (naturally). Especially close to r = 0 
since the accuracy of these points greatly affects the accuracy of all other points when doing the outward integration. 
So in radiallog.py we define rho_k = -10 + (k-1)/48 and r_k = exp(rho_k)/Z. 
"""


"""
- Make a table containing the analytical E_n = - Z^2/2n^2 energies and the energies given by radial.py and given by radiallog.py. 
  do this for the orbitals 1s, 2s, 3s, 6s, and 9s. Add a column showing the number of gridpoints used by radiallog.py. Note that
  radial.py uses 10000 points for all orbitals. (We choose to add a column showing the error as well). 
Done.

- Explain which method is more accurate and why. 
Since we are doing numerical integration (from one point to another point), the accuracy is dependent on the 
distance between these points; if the points are closer together the accuracy will be better. This is especially
important in parts where the values differ greatly from one point to another (i.e in general close to r = 0).
So since the radiallog.py file uses a logarithmic distribution of grid points, this algo has more gridpoints
close to r = 0 (i.e the distance between these are much lower). So this increases the numerical accuracy. 

- Plot the radial wave function of the orbitals in the same plot, check that the number of nodes is correct. 
"""

# Assume 1s 2s, 3s, 6s, and 9s? 
# WHY SHOULD WE HAVE P/sqrt(r) ?????????? 

# l = 0
# Z = 1
# nvals = [1,2,3,6,9]

# for n in nvals:
#     r, P, E = radial_log.radiallog(l, n, Z, plot = False)
#     plt.plot(r, P, label =f"n = {n}")

# plt.legend()
# plt.title("Radial wave functions of s orbitals, Z = 1")
# plt.xlabel("r [a.u]")
# plt.ylabel("P(r)")
# plt.xlim(0,200)
# plt.show()

# Number of nodes seem correct. 



############## QUESTION 4 ################
"""
For a few different orbitals compare the radial expectation values with the radial density functions
by plotting them in the same plot. 
"""


# # 1s orbital. 
# n = 1
# l = 0
# Z = 1
# r, P, E = radial.radial(l, n, Z, plot = False) 
# r_expec = 1.5000159390837613
# D = P**2
# plt.plot(r, D)
# plt.vlines(r_expec, -1, 1, label = "<r> = 1.50 [a.u.]", color = "r")
# plt.title("Radial density along with <r> for the 1s orbital, Z = 1") 
# plt.xlabel("r [a.u.]")
# plt.ylabel("Probability density") 
# plt.ylim(0,0.6)   
# plt.xlim(0,10)     
# plt.legend()
# plt.show() 


# #2s orbital.
# n = 2
# l = 0
# Z = 1
# r, P, E = radial.radial(l, n, Z, plot = False) 
# r_expec = 6.0001269965273583
# D = P**2
# plt.plot(r, D)
# plt.vlines(r_expec, -1, 1, label = "<r> = 6.00 [a.u.]", color = "r")
# plt.title("Radial density along with <r> for the 2s orbital, Z = 1")   
# plt.xlabel("r [a.u.]")
# plt.ylabel("Probability density") 
# plt.ylim(0,0.25)   
# plt.xlim(0,20)       
# plt.legend()
# plt.show() 

"""
- What is the meaning of <r>? 
Trivial my dude. But we can see that <r> is often shifted right of max(D). This makes sense 
since the area under D to the right of the max is greater than that to the left of the max. 
We can explain this by looking at the definition of <r> (sum over probabilities). 
"""


"""
- Compare the radial expectation values computed with radial or radiallog with the values
that you get when integrating the analytical expressions. 
"""

    

# 1s.
n = 1
l = 0
Z = 1

# radial uniform. 
r_u, P_u, E_u = radial.radial(l, n, Z, plot = False)
val_u = 1.5000159390837613 

# radial log.
r_l, P_l, E_l = radial_log.radiallog(l, n, Z, plot = False)
val_l = 1.4999999940688284

# Analytical. 
r = sympy.symbols("r")
val = sympy.integrate(4*r**3 * sympy.exp(-2*r), (r, 0, sympy.oo)) # = 3/2 = 1.5 

# 2s.
n = 2
l = 0
Z = 1

# radial uniform.
r_u, P_u, E_u = radial.radial(l, n, Z, plot = False)
val_u = 6.0001269965273583

# radial log. 
r_l, P_l, E_l = radial_log.radiallog(l, n, Z, plot = False)
val_l = 5.9999998945025474

# Analytical.
r = sympy.symbols("r")
expr = (1/2)*r**3 *sympy.exp(-r)*(1- (1/2)*r)**2 
val = sympy.integrate(expr, (r, 0, sympy.oo)) # = 6



#################################################################################################################################### PART 2
# LEON!!
# här började jag hora /Axel


# Question 5
"""
-preparatory question: 
det(...,x_i,...,x_j,...) = - det(...,x_j,...,x_i,...) i.e. the determinant is inherently antisym (se linalg 2)
det(a,a) = 0 i.e. the determinant of two equal cols is zero, which corresponds to Pauli exclusion
in the sense that if two wavefunctions (electrons) are equal, the slater det is zero.
"""

# Question 6
"sodium (Z=N=11)"
# a = 0.2683
# Z = 11
# N = 11
# l = 0
# n = 5


# # r,P,E = Mrl.radiallog(l,n,Z,N,a,plot=False)
# """
# Energy eigenvalues for configurations of n,l:

# 1s --> -4.7465371418244729e+01 a.u.
# 2s --> -1.6452069443214001e+00 a.u.
# 2p --> -5.5320826523874989e-01 a.u.
# 3s --> -1.9194457821222588e-01 a.u.
# 3p --> -1.0722887810570954e-01 a.u.
# 3d --> -5.7514291814854020e-02 a.u.
# 4s --> -7.2037368417279660e-02 a.u.
# 4p --> -4.9434446600451716e-02 a.u.
# 4d --> -3.2074662216110841e-02 a.u.
# 4f --> -3.1523877680810136e-02 a.u.
# 5s --> -3.7710084550623794e-02 a.u.

# Agrees with the table given in appendix.
# Make a table of energy values!!!
# """

# """
# Plots:
# """
# # (l, n, Z, N, a)
# R_01, P_01, E_01 = Mrl.radiallog(0,1,Z,N,a,plot=False)  #1s
# R_02, P_02, E_02 = Mrl.radiallog(0,2,Z,N,a,plot=False)  #2s
# R_12, P_12, E_12 = Mrl.radiallog(1,2,Z,N,a,plot=False)  #2p
# R_03, P_03, E_03 = Mrl.radiallog(0,3,Z,N,a,plot=False)  #3s
# R_13, P_13, E_14 = Mrl.radiallog(1,3,Z,N,a,plot=False)  #3p
# R_23, P_23, E_23 = Mrl.radiallog(2,3,Z,N,a,plot=False)  #3d

# plt.plot(R_01,P_01,label="1s")
# plt.plot(R_02,P_02,label="2s")
# plt.plot(R_12,P_12,label="2p")
# plt.plot(R_03,P_03,label="3s")
# plt.plot(R_13,P_13,label="3p")
# plt.plot(R_23,P_23,label="3d")
# plt.xlabel("r [a.u.]")
# plt.ylabel("P(r)")
# plt.xlim(0,5)
# plt.legend()
# plt.title("Radial wave functions for Na I")
# plt.show()

"""
Look at the energies for Na I in NIST and note in which order the configurations come. Look up Al III
and compare. Is the order of the configurations different, why could this be the case? 

Answer: The order IS different:
NaI:      AlIII:
2p^6 3s   2p^6 3s
2p^6 3p   2p^6 3p
2p^6 4s   2p^6 3d
2p^6 3d   2p^6 4s

Probably the fact that Al III has a heavier nucleus(?).
"""


# Question 7

a_Li = 0.29446094
Z_Li = 3
N_Li = 3
l = 0
n = 2

r,P,E = Mrl.radiallog(l,n,Z_Li,N_Li,a_Li,plot=False)

E_2s = Mrl.radiallog(0,2,Z_Li,N_Li,a_Li,plot=False)[2]
E_2p = Mrl.radiallog(1,2,Z_Li,N_Li,a_Li,plot=False)[2]
E_3s = Mrl.radiallog(0,3,Z_Li,N_Li,a_Li,plot=False)[2]
print("delta1 = ", np.abs(E_2p-E_2s)) # delta energy = E(orbital) - E(ground)
print("delta2 = ", np.abs(E_3s-E_2s))

"""
a_Li = 0.294461 
found by trial and error, optimizing for first NIST level: E_2s - E_2p = 0.0679061

doesn't produce the same exactness for level 2, see below

NIST:                   Our result:
Level 1 = 0.0679061     0.06790610956092138
Level 2 = 0.1239602     0.12601491552604915

Seems like we need a different value a for each level?
"""

"""
increasing a seems to increase the energy delta up to a certain point, then delta decreases again
delta 2 has a higher threshold before it changes
"""



def residuals(a):
    
    """
    asfdlks<df fuck you leon you will never get this
    """
    
    NIST = [0.0679061, 0.1239602, 0.1409064, 0.1425362, 0.1595267, 0.1661675] # delta values from NIST
    

    ground = Mrl.radiallog(0,2,Z_Li,N_Li,a,plot=False)[2] # Li ground state: 1s^2.2s

    l = 1
    n = 4
    state = Mrl.radiallog(l,n,Z_Li,N_Li,a,plot=False)[2]
    print("value = ", state-ground)
    return np.abs(state - ground) - NIST[5]

a0 = 0.2  # initial guess
optimized_variable, error = opt.leastsq(residuals, a0)
print("a_actual = ", optimized_variable)


# the above residual optimization gave a = 0.29446094 for level 1 and so on

"""
det enda som är kvar: Plot some wave functions in a single plot. 
"""

a_avg = 0.285735088 

E_test = Mrl.radiallog(1,2,Z_Li,N_Li,a_avg,plot=False)[2]
state = Mrl.radiallog(0,2,Z_Li,N_Li,a_avg,plot=False)[2]
E3 = Mrl.radiallog(1,3,Z_Li,N_Li,a_avg,plot=False)[2]
print("test = ", E3-state)

rgr,Pgr,Egr = Mrl.radiallog(0,2,Z_Li,N_Li,a_avg,plot=False)
r1, P1, E1 = Mrl.radiallog(1,2,Z_Li,N_Li,a_avg,plot=False)
r2, P2, E2 = Mrl.radiallog(0,3,Z_Li,N_Li,a_avg,plot=False)


plt.plot(rgr, Pgr, label="2s")
plt.plot(r1, P1, label="2p")
plt.plot(r2,P2, label="3s")
plt.xlabel("r [a.u.]")
plt.ylabel("P(r)")
plt.legend()
plt.title("Radial wave functions for Li I")
plt.xlim(0,40)
plt.show()










