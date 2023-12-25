import matplotlib.pyplot as plt 
import numpy as np
import scipy

plt.rcParams['text.usetex'] = True

A = 'sampleA.txt'
C = 'sampleC.txt'

def read(filename):
    V = []
    I = []
    f = open(filename, 'r')
    for line in f:
        v,i = line.split()
        V.append(float(v))
        I.append(float(i))
    return V, I


Va, Ia = read('sampleA.txt')
Vb, Ib = read('sampleB.txt')    # sample B, doubble barrier
Vc, Ic = read('sampleC.txt')
Vd, Id = read('sampleD.txt')


fig,(ax1, ax2) = plt.subplots(1,2)
ax1.scatter(Vc, Ic) # single barrier
ax1.set_title("Sample C")
ax1.set_xlabel("U [V]")
ax1.set_ylabel("I [A]")
ax2.scatter(Vd, Id) # single 
ax2.set_title("Sample D")
ax2.set_xlabel("U [V]")
ax2.set_ylabel("I [A]")


plt.show()


fig, (ax1, ax2) = plt.subplots(1,2)
ax1.scatter(Va, Ia) # double
ax1.set_title("Sample A")
ax1.set_xlabel("U [V]")
ax1.set_ylabel("I [A]")
ax2.scatter(Vb, Ib) # double (data provided)
ax2.set_title("Sample B")
ax2.set_xlabel("U [V]")
ax2.set_ylabel("I [A]")
plt.show()


h = 6.62607015*10**(-34) / 2*np.pi  # plancks constant
m = 0.067*9.109383710*10**(-31) # effective mass
gamma = 0.30
E = 0
e = 1.60217663 * 10**(-19)  
V0 = 200 * 10**(-3) * e    # potential height in joule

T = []

V = []
for u in Vd:
    V.append(V0 - (u*gamma*e)/2)

for val in V:     # note that sample C was broken, no barrier at all (Ohm law)
    T.append(2*np.sqrt((2*m*val) / (h**2))) ## MINUS????

L = scipy.stats.linregress(T, np.log(Id))
k = L.slope
print(k)
m = L.intercept

plt.scatter(T, np.log(Id)) # log(Id) for d, no log for c because already linear (broken)
plt.plot(T, np.multiply(T, k) + m, "orange", label="linear fit")
plt.xlabel(r"$\hat{U} \: [m^{-1}]$")
plt.ylabel(r"$\hat{I}(\hat{U})$")
plt.title("Sample D")
plt.legend()

plt.show()





