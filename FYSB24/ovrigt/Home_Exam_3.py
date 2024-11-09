import numpy as np 


E_1 = 5610.0051
E_2 = 5610.3062
E_3 = 5610.5213

A = (2/7)*(E_2-E_1)

print(A)

A = (2/5)*(E_3- E_2)

print(A)

upAs = (5/2)*A
print(upAs)

dwnAs = (7/2)*A
print(dwnAs)
SIXa = 6*A


# 1c

a = 5615.6512   # E3
b = 5615.8062   # E1
c = 5615.8303   # E6
d = 5615.9523   # E4
e = 5616.0454   # E7
f = 5616.1073   # E2
g = 5616.1674   # E5

omega = e - SIXa

print("diffs = ", f-b)

print(SIXa)
print(dwnAs)

print("AP = ", -(2/5)*(SIXa + omega - g))
print((1/6)*(f - omega - dwnAs))

# 3B

hbar = 6.62607015*10**(-34) /(2*np.pi)
k_B = 1.380649 * 10**(-23) 
tau = 27*10**(-9)

T = hbar/(2*k_B*tau)
print(T)

M = 85.4678 * 1.66053907 * 10**(-27)

v_mp = np.sqrt((2*k_B*T)/M)
v_rms = np.sqrt((3*k_B*T)/M)

print("root mean square = ", v_rms)
print("most probable = ", v_mp)

lamb = 780 * 10 **(-9)
tau = 27 * 10 **(-9)
c = 300000000

del_lamb = lamb**2 /(2*np.pi*tau*c)
print(del_lamb / 10**(-9))

