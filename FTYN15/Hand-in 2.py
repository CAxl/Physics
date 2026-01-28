import numpy as np 
import matplotlib.pyplot as plt 
from uncertainties import ufloat, ufloat_fromstr
from uncertainties.umath import *  # sin(), etc.
from uncertainties import unumpy
import numpy as np
from pylab import *
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset



# all spins initially up

H = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])


# std : ["(0.0014)", "(0.0015)", "(0.0023)", "(0.0020)", "(0.0018)", "(0.0015)", "(0.0014)"]
m_su = np.array([-0.9921, -0.9907, 0.9832, 0.9866, 0.9889, 0.9907, 0.9920])

plt.scatter(H, m_su)

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.xlabel("H [arb. units]")
plt.ylabel("m [arb. units]")
plt.title("s_i intitially up")
plt.grid()
plt.show()


plt.title("s_i initially up; zoomed")
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.scatter(H, m_su)
plt.xlabel("H [arb. units]")
plt.ylabel("m [arb. units]")
plt.grid()
plt.tick_params(direction='in')
plt.ylim((0.97,1))
plt.xlim((-0.15, 0.35))
plt.show()

# all spins initially down

# std = [(0.0014), (0.0015), (0.0018), (0.0021), (0.0025), (0.0016), (0.0014)]
m_sd = np.array([-0.9922 , -0.9907 , -0.9888 , -0.9865 , -0.9831 , 0.9905 , 0.9921 ])



plt.scatter(H,m_sd)
plt.xlabel("H [arb. units]")
plt.ylabel("m [arb. units]")
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.title("s_i initially down")
plt.grid()
plt.show()


plt.title("s_i initially down; zoomed")
plt.scatter(H,m_sd)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.xlabel("H [arb. units]")
plt.ylabel("m [arb. units]")
plt.grid()
plt.tick_params(direction='in')
plt.ylim((-1,-0.97))
plt.xlim((-0.25, 0.2))


plt.show()



# exercise 2
# std = [(0.0010), (0.0018), (0.0024), (0.0038), (0.0040), (0.0076), (0.0057), (0.0046), (0.0074)]

m = np.array([-0.99, -0.985, -0.98, -0.975, -0.97, -0.965, -0.96, -0.955, -0.95])
E = np.array([-1.9625 , -1.9463 , -1.9311 , -1.9189 , -1.9103 , -1.9034 , -1.8982 , -1.9149 , -1.9192 ])
rho = [(1+i)/2 for i in m]
rho = np.array(rho)

def func(rho):
    return 8*rho - 2

plt.scatter(rho, E)
plt.grid()
plt.title("Random initial state")
plt.xlabel("rho [number density]")
plt.ylabel("E [arb. units]")
#plt.plot(rho,func(rho), '--', color="black", label="-2 + 8rho")
#plt.legend()
plt.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
plt.show()



# circular initial state
# std = [(0.0009), (0.0014), (0.0029), (0.0036), (0.0041), (0.0044), (0.0079), (0.0065), (0.0046)]
E = np.array([-1.9660 , -1.9498 , -1.9347 , -1.9201 , -1.9120 , -1.9271 , -1.9305 , -1.9325 , -1.9319 ])

plt.scatter(rho, E)
plt.title("Circular initial state")
plt.xlabel("rho [number density]")
plt.ylabel("E [arb. units]")
#plt.plot(rho, func(rho), '--', color="black", label="-2+8rho")
#plt.legend() 
plt.grid()
plt.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
plt.show()



dE = [0.0009, 0.0014, 0.0029, 0.0036, 0.0041, 0.0044, 0.0079, 0.0065, 0.0046]
dr = [0, 0, 0, 0, 0, 0,0 ,0,0]

fig, ax = plt.subplots(1)
## plot points with errors in both axes
plt.errorbar(rho, E, yerr=dE, xerr=dr, ecolor='black', fmt='o', label='data')
plt.scatter(rho,E)
## plot line corresponding to fit

plt.legend(loc=2)

# create zoomed region
x1 = 10.7 + 0.001
x2 = 10.7 - 0.001
# select y-range for zoomed region
y1 = 5.966
y2 = 5.964
# Make the zoom-in plot:
axins = zoomed_inset_axes(ax, 1000, loc=4) # zoom = 2
axins.scatter(rho, E,'r')

axins.errorbar(x, y, yerr=dy, xerr=dx, ecolor='black', fmt='o', label='data')
axins.set_xticklabels([])
plt.xticks(visible=False)
plt.yticks(visible=True)
plt.tick_params(direction="in",top="True",right="True")
mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")
plt.draw()

plt.show()





