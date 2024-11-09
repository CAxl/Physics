import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize

c1 = 15.9
c2 = 18.4
c3 = 0.71
c4 = 23.2
c5 = 11.5

m_n = 939.565
m_hydrogen = 938.783
delta = 1   # even-even

def B(A,Z): # define binding energy
    N = A-Z
    return c1*A - c2*(A**(2/3)) - c3*(Z**2)*(A**(-1/3)) - c4*((N-Z)**2)/A + c5*delta*(A**(-1/2))

def M(Z,A): # Atomic mass 
    N = A-Z
    return Z*m_hydrogen + N*m_n - B(A,Z)

A = [2*i for i in range(1,40)]  # list of even constant A values

#soln = minimize(M, 10, args=(62))
#print(soln.x[0])

Zlist = []
for val in A:
    soln = minimize(M, 10, args=(val)) # for each A, find Z which minimize M(A,Z)
    Zlist.append(soln.x[0]) 

Nlist = [A[i] - Zlist[i] for i in range(0,39)]

plt.scatter(Nlist,Zlist)
plt.xlabel("N")
plt.tick_params(direction='in', top=True, right=True)
plt.ylabel("Z")
plt.legend()
#plt.show()



############### 1b ###################

def Bin(N,Z):   # binding energy
    A = N + Z
    if(N%2==0):
        return c1*A - c2*(A**(2/3)) - c3*(Z**2)*(A**(-1/3)) - c4*((N-Z)**2)/A + c5*delta*(A**(-1/2))
    else:
        return c1*A - c2*(A**(2/3)) - c3*(Z**2)*(A**(-1/3)) - c4*((N-Z)**2)/A

def s_n(Z, N): # neutron drip line
    return Bin(N,Z) - Bin(N-1,Z)    # we step down N with 1, but for odd-even change paring param(above)

def s_p(Z, N): # proton drip line
    return Bin(N,Z) - Bin(N,Z-1)


def Ndrip(Z):
    #N = [2*i for i in range(1,100)] # even number of neutrons
    nlist = []

    for i in Nlist:
        if(s_n(Z,i)>0):
            nlist.append(s_n(Z,i))
        else:
            break
    ind = nlist.index(nlist[-1])
    return Nlist[ind]
    
def Pdrip(Z):
    #N = [2*i for i in range(1,100)] # even number of nuetrons
    plist = []
    for i in Nlist:
        plist.append(np.abs(s_p(Z,i)))
    ind = plist.index(plist[-1])
    return plist[ind]

N_line = []
P_line = []
for i in Zlist:
    N_line.append(Ndrip(i))
    P_line.append(Pdrip(i))
#print(N_line)
print(P_line)

plt.plot(N_line, Zlist)

plt.show()


