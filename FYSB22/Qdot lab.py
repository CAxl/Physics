import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# plt.rcParams['text.usetex'] = True

# well_data = pd.read_csv("quantum well 1.csv", delimiter = ";", decimal=",")
# v = "Våglängd (nm)"
# I = "Intensitet Källa #1"

# plt.plot(well_data[v][708:1620], well_data[I][708:1620])
# plt.xlabel(r"Wavelength [nm]")
# plt.ylabel(r"Intensity")
# plt.title(r"Quantum wells")

# Vpeak1 = well_data.loc[well_data[I][800:900].idxmax()][v]
# Ipeak1 = well_data.loc[well_data[I][800:900].idxmax()][I]
# Vpeak2 = well_data.loc[well_data[I][900:996].idxmax()][v]
# Ipeak2 = well_data.loc[well_data[I][900:996].idxmax()][I]
# Vpeak3 = well_data.loc[well_data[I][996:1070].idxmax()][v]
# Ipeak3 = well_data.loc[well_data[I][996:1070].idxmax()][I]
# Vpeak4 = well_data.loc[well_data[I][1070:1130].idxmax()][v]
# Ipeak4 = well_data.loc[well_data[I][1070:1130].idxmax()][I]
# print(Vpeak1, Vpeak2, Vpeak3, Vpeak4 )

# plt.annotate(r"$\lambda_1$", xy= (Vpeak1, Ipeak1 + 2), xytext =(Vpeak1, Ipeak1 + 10), arrowprops = dict(facecolor ='green'))
# plt.annotate(r"$\lambda_2$", xy= (Vpeak2, Ipeak2 + 2), xytext =(Vpeak2, Ipeak2 + 10), arrowprops = dict(facecolor ='green'))
# plt.annotate(r"$\lambda_3$", xy= (Vpeak3, Ipeak3 + 2), xytext =(Vpeak3, Ipeak3 + 10), arrowprops = dict(facecolor ='green'))
# plt.annotate(r"$\lambda_4$", xy= (Vpeak4, Ipeak4 + 2), xytext =(Vpeak4, Ipeak4 + 10), arrowprops = dict(facecolor ='green'))
# plt.ylim(0, 110)
# plt.show()


# Gdot = pd.read_csv("quantum dot green sample.csv", delimiter = ";", decimal=",")
# Odot = pd.read_csv("quantum dot orange sample.csv", delimiter = ";", decimal=",")

# fig,(ax1, ax2) = plt.subplots(1,2)

# ax1.plot(Gdot[v][157:1308], Gdot[I][157:1308])
# Gpeak = Gdot.loc[Gdot[I][157:1308].idxmax()][v]
# labels = ["400", "500", r"$\lambda_G$", "600", "700", "800"]
# ax1.set_xticks([400, 500, Gpeak, 600, 700, 800], labels)
# ax1.set_xlabel(r"Wavelength [nm]")
# ax1.set_ylabel(r"Intensity")
# ax1.set_title("Green sample")

# I2 = "Intensitet Källa #2"
# ax2.plot(Odot[v][157:1308], Odot[I2][157:1308])
# Opeak = Odot.loc[Odot[I2][157:1308].idxmax()][v]
# labels = ["450", "550", r"$\lambda_O$", "650", "750"]
# ax2.set_xticks([450, 550, Opeak, 650, 750], labels)
# ax2.set_xlabel(r"Wavelength [nm]")
# ax2.set_ylabel("Intensity")
# ax2.set_title("Orange sample")
# ax2.set_ylim(0,60)

# print(Gpeak, Opeak)

# plt.show()


me = 0.13 * 9.1093837 * 10**(-31)
mh = 0.45 * 9.1093837 * 10**(-31)
hb = 6.62607015 * 10**(-34) / (2*np.pi)
c = 299792458
lamdaG = 548 * 10**(-9)
lamdaO = 604 * 10**(-9)
e = 1.60217663 * 10**(-19)
Eg = 1.74 * e 


L = np.sqrt( (2.2 * (hb**2) * (np.pi**2) / (2)) * ( (1/me) + ( 1/mh ) ) * ( (c * hb * 2*np.pi)/(lamdaO) - Eg )**(-1) )
print(L)



