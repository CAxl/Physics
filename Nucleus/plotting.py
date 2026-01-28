from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy import integrate


def plot_nuc(filename, func_label = "", dirpath = "C:/Dev/Nucleus/Output/"):

    path = os.path.join(dirpath, filename)
    xy_data = np.loadtxt(path)
    plot = plt.plot(xy_data[:,0], xy_data[:,1], label = func_label)
    return plot


def plot_radial_wvfnc(filename, func_label = "", dirpath = "C:/Dev/Nucleus/Output/"):
    path = os.path.join(dirpath, filename)
    xy_data = np.loadtxt(path)
    plot = plt.plot(xy_data[:,0], xy_data[:,1]/xy_data[:,0], label = func_label)    # plot R(r) = u(r) / r
    return plot


# plot the potentials
plot_nuc("V_WS+V_C", "V_WS+V_C")
plot_nuc("V_WS", "V_WS")
plt.xlabel("r")
plot_nuc("V_C", "V_C")
plt.legend()
plt.show()

dirpath = "C:/Dev/Nucleus/Output/"
V_Cvals = np.loadtxt(os.path.join(dirpath, "V_C"))[:,1]
print(V_Cvals) # weird that V_C is 5 at r=r_0??


# ---------------- testing the norm function --------------------------
# xy = np.loadtxt("C:/Dev/Nucleus/Output/p_norm_test.txt")


# xlist = xy[:,0]
# ylist = xy[:,1]

# # plt.plot(xlist, ylist)
# # plt.show()
# integ = integrate.simpson(ylist**2, x=xlist)    # not sure if this works 17/10-25
# print(integ)

# ----------------------------------------------------------------------



# plot_nuc("p_wf_p3_2_1.txt", "proton 0p_3/2")
# plot_nuc("p_wf_s1_2_1.txt", "proton 0s_1/2")
# plot_nuc("p_wf_p1_2_1.txt", "proton 0p_1/2")
# plt.legend()
# plt.title("Radial wavefunctions u_nlj(r) for 16O")
# plt.xlabel("r [fm]")
# plt.ylabel("u(r) [arb. units]")
# plt.show()

# plot_nuc("p_Rwf_s1_2_1.txt")
# plt.show()


# plot_nuc("p_NormRwf_s1_2_1.txt")
# plt.show()

