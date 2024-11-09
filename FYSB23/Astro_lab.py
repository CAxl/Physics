import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import scipy as sp

plt.rcParams['text.usetex'] = True

k_B = 1.380649 * 10**(-23)
e = 1.60217663 * 10**(-19)  

spect = pd.read_csv("C:\Dev\Physics\FYSB23\spectrum.dat", delim_whitespace=True, decimal=".")
lamb = "lambda/AA"
F = "F/Fc"

lamb1_index = spect.index.get_loc(spect[spect[lamb] == 6392.0010].index[0])
lamb2_index = spect.index.get_loc(spect[spect[lamb] == 6402.0009].index[0])

plt.plot(spect[lamb][lamb2_index:lamb1_index], spect[F][lamb2_index:lamb1_index])
plt.xlabel("Wavelength [Å]")
plt.ylabel("Intensity")
plt.title("Choosing prominent iron peaks")
plt.show()

val = np.log10(np.e)/k_B
#print(val*e)    # e/k_B = 5039.778172753161 [eV]



# ################ 3 ####################

# a = spect.index.get_loc(spect[spect[lamb] == 6393.2604].index[0])
# b = spect.index.get_loc(spect[spect[lamb] == 6394.0012].index[0])

# plt.plot(spect[lamb][b:a], spect[F][b:a])
# plt.show()


# lamb_min = 6393.2604
# lamb_max = 6394.0012
def integral(min, max):
    i = 0
    F_list = []
    lamb_list = []
    for val in spect[lamb]:
        if val > min and val < max:
            F_list.append(1 - spect[F][i])      # return 1 - F[i] (integrand)
            lamb_list.append(spect[lamb][i])    # return corresponding lambda value
        i += 1
   
    return np.abs(sp.integrate.simpson(F_list, lamb_list))  # integral; function F_list over lambda

# print(integral(lamb_min, lamb_max))


############### 4 #################

upper_index = spect.index.get_loc(spect[spect[lamb] == 6392.0333].index[0])
lower_index = spect.index.get_loc(spect[spect[lamb] == 6577.2704].index[0])
plt.plot(spect[lamb][lower_index:upper_index], spect[F][lower_index:upper_index])
plt.xlabel(r"Wavelength [Å]")
plt.ylabel(r"Intensity")
plt.title(r"Sun spectrum")
plt.show()


p1min = 6393.287
p1max = 6394.018
p1 = integral(p1min,p1max)

p2min = 6416.682
p2max = 6417.170
p2 = integral(p2min, p2max)

p3min = 6430.572
p3max = 6431.170
p3 = integral(p3min, p3max)

p4min = 6481.726
p4max = 6482.081
p4 = integral(p4min, p4max)

p5min = 6494.678
p5max = 6495.299
p5 = integral(p5min, p5max)

p6min = 6496.255
p6max = 6496.682
p6 = integral(p6min, p6max)

p7min = 6498.771
p7max = 6499.123
p7 = integral(p7min, p7max)

p8min = 6518.1945
p8max = 6518.5514
p8 = integral(p8min, p8max)

p9min = 6533.7277
p9max = 6534.1455
p9 = integral(p9min, p9max)

p10min = 6545.988
p10max = 6546.536
p10 = integral(p10min, p10max)

p11min = 6574.0731
p11max = 6574.3978
p11 = integral(p11min, p11max)


int_list = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]
print("S list = ",int_list)
loggf = [-1.452, -1.109, -2.005, -2.981, -1.268, -0.530, -4.687, -2.438, -1.360, -1.536, -5.004]
chi = [2.433, 4.796, 2.176, 2.279, 2.404, 4.796, 0.958, 2.832, 4.559, 2.759, 0.990]
lambs = [6393.6004, 6416.9331, 6430.8450, 6481.8698, 6494.9804, 6496.4656, 6498.9383, 6518.3657, 6533.9281, 6546.2381, 6574.2266]


i = 0
func = []
for val in int_list:
    func.append(np.log10(val/lambs[i]) - loggf[i] - np.log10(lambs[i]))
    i+=1
plt.scatter(chi, func)
a,b = np.poly1d(np.polyfit(chi,func,1))
line = np.linspace(0,7,100)
plt.plot(line,a*line+b, color='orange', linestyle="--",label=f"a = {np.round(a,2)}")
plt.xlabel(r"Excitation potential [eV]")
plt.ylabel(r"LHS Equation (8)")
plt.title(r"Line strength versus excitation potential")
plt.legend()
plt.show()
print(a)    # a = slope of linear fit

T_inv = func/a
print(1/T_inv)


"""
felkällor: den givna datan \sim 10% (at worst)
normalized spectra (Not where we're looking)!!

"""


val = np.log10(np.e)/k_B
print(val*e)                # e/k_B = 5039.778172753161 [eV]

T_in = np.abs(a)/(val*e)
print("Temp = ",1/T_in)     # print temp in kelvin


de = 5772 - (1/T_in)
print(de/5772*100)


