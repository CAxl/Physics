import analytical_solution as AS  
import radial as rd
import radial_log as rlog
import numpy as np 
import matplotlib.pyplot as plt 

# Questions: 
# What is the radial wave function P(r)?
    # P(r) = rR(r) is a separated factor in the common eigenfunction for H, l^2 and l_z (only dependent on r)
# What is the radial density function D(r) and how do you interpret it?
    # D(r) integrates over the area of the harmonics to give us a one dimensional measure of the 
    # prabability to find an electron at a certain distance r from the center. D(r) takes for every r
    # the sum of the probabilities over the spherical harmonics' area. Therefore we integrate (summation)
    # summation over the angular probability density (harmoniics) return for every measure of r
    # a probability density, thus we have a function only dep on r --> prob to find electron.


# Question 1
r = np.linspace(0,30,1000)

Z = 1
D3s = AS.P3s(r,Z)**2
D3p = AS.P3p(r,Z)**2
D3d = AS.P3d(r,Z)**2    # D(r) = P(r)**2 * 1

plt.plot(r,D3s, label="D(r) for 3s")
plt.plot(r,D3p, label= "D(r) for 3p")
plt.plot(r,D3d, label="D(r) for 3d")
plt.xlabel("r")
plt.ylabel("Probability density")
plt.legend()
plt.show()


# Number of nodes is given by v = n-l-1, checks out

# Question 2
r=np.linspace(0,10,1000)
def V(r,l): 
    return -Z/r + (l*(l+1))/(2*r**2)


plt.ylim(-1.5,1.5)
plt.plot(r,V(r,0), label="Coulomb potential (l=0)")
plt.plot(r,V(r,1), label="l=1")
plt.plot(r,V(r,2), label="l=2")
plt.xlabel("r")
plt.ylabel("V(r) = -Z/r + l(l+1)/2r^2")
plt.grid()
plt.show()


# Question 3
l = 0
n = 9
Z = 1
#r, P, E = rd.radial(l,n,Z)
r, P, E = rlog.radiallog(l,n,Z)



# Question 4


# --------------------------------------------------------------------
# part 2

# prep question
# determinants have det(x_i,x_j) = -det(x_j,x_i)
# det(two equal cols) = 0





