"""
@author:
Axel Cerne
March 2024
"""

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

plt.rcParams['text.usetex'] = True

c1 = 15.9
c2 = 18.4
c3 = 0.71
c4 = 23.2
c5 = 11.5

hbar = 6.62607015e-34 / (2*np.pi)
c = 299792458


m_n = 939.565 # [MeV]
m_hydrogen = 938.783 # [MeV]
delta = 1   # even-even

def B(A,Z): # define binding energy
    N = A-Z
    return c1*A - c2*(A**(2/3)) - c3*(Z**2)*(A**(-1/3)) - c4*((N-Z)**2)/A + c5*delta*(A**(-1/2))

def M(Z,A): # Atomic mass 
    N = A-Z
    return Z*m_hydrogen + N*m_n - B(A,Z)


A = [2*i for i in range(1,40)]  # list of even constant A values
Zlist = []
for val in A:
    soln = minimize(M, 10, args=(val)) # for each A, find Z which minimize M(A,Z)
    Zlist.append(soln.x[0]) 

plt.scatter(A,Zlist)
plt.hlines(28, 0, 62, color='red', linestyles='--', label=f"Z={28}")
plt.vlines(62,0,28, color='black', linestyles='--', label=f"A={62}")
plt.xlabel("Mass number A")
plt.tick_params(direction='in', top=True, right=True)
plt.ylabel("Proton number Z")
plt.title("Z values minimizing M(A,Z)")
plt.legend()
plt.show()


############### 1b ###################
def B(A,Z):   # binding energy, if even-even --> delta = 1, if even-odd --> delta = 0, if odd-odd --> delta = -1
    if((A-Z)%2==0 and (Z%2==0)):
        return c1*A - c2*(A**(2/3)) - c3*(Z**2)*(A**(-1/3)) - c4*(((A-Z)-Z)**2)/A + c5*(A**(-1/2))
    elif((A-Z)%2!=0 or (Z%2!=0)):
        return c1*A - c2*(A**(2/3)) - c3*(Z**2)*(A**(-1/3)) - c4*(((A-Z)-Z)**2)/A
    elif((A-Z)%2!=0 and (Z%2!=0)):
        return c1*A - c2*(A**(2/3)) - c3*(Z**2)*(A**(-1/3)) - c4*(((A-Z)-Z)**2)/A - c5*(A**(-1/2))

def s_n(A): # neutron drip line
    Z = 28
    return B(A,Z) - B(A-1,Z)    # we step down N with 1, but for odd-even change paring parameter(above)

def s_p(A): # proton drip line
    Z = 28
    return B(A,Z) - B(A-1,Z-1)  # step down Z with 1 => A --> A-1
# when we remove N the binding energy of the isotope goes down, when it has gone down enough that B(A,28) = B(A,27), the nucleus decays by proton emission

A = np.arange(2,150,1)

sn = [s_n(i) for i in A]    # calculate separation energy for each A
ind = sn.index(min(n for n in sn if n>0))   # collect smallest positive value for separation energy
print("A(neutron drip) = ", A[ind]-28)

sp = [s_p(i) for i in A]    
#print(sp)
ind = sp.index(min(n for n in sp if n>0))   # collect smallest positive value for separation energy
print("A(proton drip) = ", A[ind-1]-28)    # collect value before Z=27 allowed


# ########### problem 2 ################

hbar = 6.62607015e-34 / (2*np.pi)
c = 299792458 

A1 = [11, 15, 19, 23, 29, 35, 41, 45]  # Mirror nuclei considered
A2 = [pow(A,2/3) for A in A1]  # range over which we plot \Delta E

M1atom = [11011432.60, 15003065.6, 19001880.91, 22994123.77, 28981800.4, 34975257.7, 40969251.16, 44965768.5]   # atomic masses [\mu AMU] proton rich
M2atom = [11009305.167, 15000108.8983, 18998403.1621, 22989769.2820, 28976494.6643, 34968852.69, 40962277.91, 44958120.8]   # atomic masses [\mu AMU] neutron rich

# convert atomic mass to nuclear mass
Z1 = np.array([6, 8, 10, 12, 15, 18, 21 ,23])
Z2 = np.array([5, 7, 9, 11, 14, 17, 20, 22])
m_e = 9.1093837 * 1e-31
electronmasses1 = m_e*Z1
electronmasses2 = m_e*Z2
M1 = M1atom - electronmasses1
M2 = M2atom - electronmasses2

# convert mass difference to energy difference in [J] or [MeV]
diff = np.subtract(M1, M2) * 1e-6 * 1.66053907*1e-27 *c**2 * 6.24*1e12

# atomic masses proton-rich nuclei [\mu AMU]
y_i1 = [
    ufloat_fromstr("11011432.60(6)"),   #A=11
    ufloat_fromstr("15003065.6(5)"),    #A=15
    ufloat_fromstr("19001880.91(17)"),  #A=19
    ufloat_fromstr("22994123.77(3)"),   #A=23
    ufloat_fromstr("28981800.4(4)"),    #A=29
    ufloat_fromstr("34975257.7(7)"),    #A=35
    ufloat_fromstr("40969251.16(8)"),   #A=41
    ufloat_fromstr("44965768.5(9)")     #A=45
]
# atomic masses neutron-rich nuclei [\mu AMU]
y_i2 = [
    ufloat_fromstr("11009305.167(13)"),
    ufloat_fromstr("15000108.8983(6)"),
    ufloat_fromstr("18998403.1621(9)"),
    ufloat_fromstr("22989769.2820(19)"),
    ufloat_fromstr("28976494.6643(6)"),
    ufloat_fromstr("34968852.69(4)"),
    ufloat_fromstr("40962277.91(15)"),
    ufloat_fromstr("44958120.8(9)")
]
Y1 = np.array(y_i1)
Y2 = np.array(y_i2)
Delta_E = np.subtract(Y1,Y2) * 1e-6 * 1.66053907*1e-27 *c**2  * 6.24*1e12  # convert to SI (or MeV)

print("_______________________")
print(Y1 * 1e-6 * 1.66053907*1e-27 * c**2 * 6.24*1e12)  # print statements to collect errors
print(Y2 * 1e-6 * 1.66053907*1e-27 * c**2 * 6.24*1e12)
print("_______________________")
print(Delta_E)
print("_______________________")

y  = [ i.n for i in Delta_E ] # n -> nominal value
dy = [ i.s for i in Delta_E ] # s -> standard deviation.

A_vals = [ufloat(A,abs(A)*sys.float_info.epsilon) for A in A2]

x  = [ i.n for i in A_vals ] # n -> nominal value
dx = [ i.s for i in A_vals ] # s -> standard deviation


def func(p, x):
    "function to fit"
    a, b = p
    return a + b*x

# Model object
quad_model = odr.Model(func)

# Create a RealData object
data = odr.RealData(x, y, sx=dx, sy=dy)

# Set up ODR with the model and data.
odr = odr.ODR(data, quad_model, beta0=[0., 1.]) # initial guess of parameters

# Run the regression.
out = odr.run()

#print fit parameters and 1-sigma estimates
popt = out.beta
perr = out.sd_beta
print('fit parameter 1-sigma error')
print('———————————–')
for i in range(len(popt)):
    print("{:6.2f} +/- {:6.2f}".format(popt[i], perr[i]))
print()
    
a = ufloat(popt[0],perr[0])
b = ufloat(popt[1],perr[1])
print("a = {:uS}".format(a))
print("b = {:uS}".format(b))

# prepare confidence level curves
nstd = 2. # to draw 2-sigma intervals
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr

x_fit = np.linspace(min(x), max(x), 100)
fit = func(popt, x_fit)
fit_up = func(popt_up, x_fit)
fit_dw = func(popt_dw, x_fit)

#plot
fig, ax = plt.subplots(1)
plt.xlabel(r"Mass number A$^{2/3}$")
plt.ylabel(r"$\Delta$ E [MeV]")
plt.title(r'Linreg. for $\Delta E$ vs $A^{2/3}$')
## plot points with errors in both axes
plt.errorbar(x, y, yerr=dy, xerr=dx, ecolor='black', fmt='o', label='data')
#plt.scatter(A2,diff)
## plot line corresponding to fit
plt.plot(x_fit, fit, 'r', lw=0.5, label='best fit curve')
## color a 5 sigma region to the fit 
ax.fill_between(x_fit, fit_up, fit_dw, alpha=.25, label='2-sigma interval',color='teal')
plt.legend(loc=2)

# create zoomed region
x1 = 10.7 + 0.001
x2 = 10.7 - 0.001
# select y-range for zoomed region
y1 = 5.966
y2 = 5.964
# Make the zoom-in plot:
axins = zoomed_inset_axes(ax, 1000, loc=4) # zoom = 2
axins.plot(x_fit, fit,'r')
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.scatter(A2,diff)
axins.fill_between(x_fit, fit_up, fit_dw, alpha=.25, label='5-sigma interval', color='teal')
axins.errorbar(x, y, yerr=dy, xerr=dx, ecolor='black', fmt='o', label='data')
axins.set_xticklabels([])
plt.xticks(visible=False)
plt.yticks(visible=True)
plt.tick_params(direction="in",top="True",right="True")
mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")
plt.draw()

plt.show()
print("b=", b)  # collect linear coefficient

r0 = 1e15*(3*hbar*c)/(5*137*b)  # calculate r0 [fm]
print("r0 = ", r0," [fm]")

# LIST = np.array([pow(A,1/3) for A in A1])   # check Radii of the given isotopes
# print("final R = ", r0*LIST, "[fm]")


################## 3 ##########################

delE1 = 0.752
delE2 = 1.106
delE3 = 1.587

I0 = 3/delE1
I2 = 7/delE2
I4 = 11/delE3

avg = (I0 + I2 + I4)/3
print("I_avg = ", avg, "hbar/MeV")

r0 = 1.2
A = 48
R = r0*A**(1/3) / 197.3 # MeV^-1
M = 48 * 931.5
Irigid = (2/5) * M * R**2 * (1 + (0.31*0.35))    # 1u = 931.5 MeV/c^2

print("rigid =", Irigid)
print(avg/Irigid)

# 3e
e = 1.60217663 * 1e-19

omega = (1.106/2)*e*1e6/hbar  # 1/s
print("omega = ", omega, "1/s")
T = 2 * np.pi / omega
print("Period = ",T)

M = 1.66053907 * 1e-27 # kg

R = r0*A**(1/3)*1e-15 #[m]
v_c = np.sqrt((2* 35 * e *1e6) / M)
T_c = 2*R / v_c
print("T = ", T)
print("T_classic = ", T_c)


################## 4 ###################

#a 
E = ufloat_fromstr("190(5)") * 1e6 * 1.60217663 * 1e-19  # E per fission [J]
P = 2.2*1e9 # power in [W]

R_f = P/E

M = 235*1.66053907*1e-27 #[kg]
sec_per_year = 31556926

prob = ufloat_fromstr("0.840(2)")
Mtot = R_f*sec_per_year*M/prob

#print(Mtot, "kg per year")

# b
R = ufloat_fromstr("0.0040(1)")
N_235 = Mtot/M

N_238 = N_235*R*0.997/0.0072
M_pu = 239*1.66053907*1e-27

Mpu = N_238*M_pu
print("production of Pu = ", Mpu, "kg/year")

#c
E = ufloat_fromstr("225(5)") * 1e6 * 1.60217663 * 1e-19
T = ufloat_fromstr("60.5(2)") * 24 * 60 * 60

m_fiss = 0.5*1e-9
Mcf = 254*1.66053907*1e-27

N = m_fiss/Mcf
Rf = N/T

P = Rf*E
print("power = ", P, "J/s")

# d
Cv = ufloat_fromstr("116(60)")
print("Cv = ", Cv)
mass = 1e-9 # 1\mu g to kg
delT = P/(60*mass*Cv)
print("delta T = ", delT, "K/min")



