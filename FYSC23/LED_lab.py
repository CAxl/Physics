from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"})

# Part 1


"""
Current vs input voltage.
Setup was 

Which wavelength: 420 - 700 nm, peak 1: 455.31 \pm 0.3 nm, peak 2: 581.74 \pm 0.3 nm
(non uniform)

phospurus coating (wrapped around blue LED) takes blue LED light absorbes it and emits a wide range of secondary emission. (thus one peak at blue, then a broad spectrum of random shit)
Light emission not usiform because \emph{it is} a blue LED, blue comes through in peak 1

Current on voltage:
"""
V = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])  # input V (V) \pm 0.1 (sensitive dial)
I = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.9, 1.7, 2.5, 3.5]  # current through LED (mA) \pm 0.1




# Define exponential function
def exp_func(x, a, b):
    return a * (np.exp(b * x) - 1)

# Fit the curve
params, covariance = curve_fit(exp_func, V, I)
a_fit, b_fit = params

# Generate fitted y values
x_fit = np.linspace(min(V), max(V), 1000)
y_fit = exp_func(x_fit, a_fit, b_fit)
print(a_fit)

# Plot data and fitted curve


plt.scatter(V, I, label='Data')
plt.plot(x_fit, y_fit, label=f'Fit: {4.41*1e-6} * [exp({b_fit:.2f} * V) - 1]', color='orange', linestyle='dashed')
plt.xlabel(r"Input voltage $V_{in}$ [V]")
plt.ylabel(r"Current $I$ [mA]")
plt.legend()
plt.tick_params(direction="in", top=True, right=True)


plt.show()

"""
We can explain the above relationship, see Book,

We cant yet explain why there is a threshhold, i.e. we think a small current should immedietly offset the equilibrium I_drift - I_diff = 0

OKay -> answ: threshold voltage is due to resistance in circuit, we see no voltage over the LED until it lights up. (according to lab supervisor, we looked and this was not true)
"""


"""
In reverse bias the I_drift is still there, constant but almost zero. -> no light emitted from LED
"""

# intensity vs current at lambda = 455.31
Current = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])    # mA \pm 0.1
Intensity = np.array([200, 1100, 2100, 3100, 3900, 4600, 5500, 6200, 6900, 7600, 8200, 8800])  # unitless \pm 100 (first must be background)


# polyfit

plt.scatter(Current, Intensity, label='Data')
poly = np.polyfit(Current, Intensity, deg=1/2, rcond=None, full=False, w=None, cov=False)
print(poly)
xs = np.linspace(min(Current), max(Current), 1000)
ys =  poly[0] * xs ** (1/2)
plt.plot(xs, ys, label=f'')
plt.legend()
plt.show()

# linear fit

# def lin_f(x,a,b):
#     return a * x + b

# params, covariance = curve_fit(lin_f, Current, Intensity)
# a_fit, b_fit = params

# x_fit = np.linspace(min(Current), max(Current), 1000)

# y_fit = lin_f(x_fit, a_fit, b_fit)

# plt.plot(x_fit, y_fit, label=f'Fit: {a_fit:.2f}'  r" * $I$"  f" + {b_fit:.2f}", color='orange', linestyle='dashed')
# plt.scatter(Current, Intensity, label="Data")
# plt.xlabel(r"Current $I$ [mA]")
# plt.legend()
# plt.ylabel(r"Intensity [counts]")
# plt.show()



"""
We can explain why there is a linear relationship
"""


# Voltage over LED vs intensity
V = np.array([2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])       # \pm 0.1
Int = [200, 200, 200, 200, 200, 200, 300, 1000, 4400, 8400]   # \pm 100

x_fit = np.linspace(min(V), max(V), 1000)

params,covariance = curve_fit(exp_func, V, Int)
a,b = params
y_fit = exp_func(x_fit, a, b)
print(a)


plt.plot(x_fit, y_fit, label=f'Fit: {2.10:.2f}e-6 * [exp({b:.2f} * V) - 1]', color='orange', linestyle='dashed')
plt.scatter(V, Int, label='Data')
plt.xlabel(r"Voltage over LED $V_{LED}$ [V]")
plt.ylabel("Intensity [counts]")
plt.legend()
plt.show()

"""
Interpretation TBD but looks similar to input Voltage
"""



# Part 2: Temperature dependence

"""
 NOTE: 100 Ohm resistor used in circuit.
 

"""

# before

wavelength = 593.4 # \pm 0.1
Input_V = 2.3 # V \pm 0.1
Current = 4.48 # mA \pm 0.001
Intensity = 49400  # unitless \pm 100
V_LED = 1.91 # V \pm 0.01

 
# after
wavelength_after = 569.7 
Input_V_after = 2.3 
Current_after = 0.660 
Intensity_after = 64600 
V_LED_after = 2.34 


# #e = 1.667 * 1e-19
# e = 1
# V = 2.3
# #k = 1.38 * 1e-23
# k=1
# T = np.linspace(1, 10, 1000) 
# Eg = 2.14	# of course this should also depend on T, but this calculation is for \lambda = 580, 
# 			# which is between \lambda_before and \lambda_after

# def I(T):
# 	I0 = (T**3) * np.exp(-Eg/(k*T))
# 	I = I0 * (np.exp(e*V/(k*T)) - 1)
# 	return I

# plt.plot(T,I(T))
# plt.xlabel("Temperature T")
# plt.ylabel("Current I(T)") 
# plt.show()



