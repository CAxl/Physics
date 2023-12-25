import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy


## Define function to convert dBm to Watt#######
def conv(P_dbm):
    P_w=10**((P_dbm)/10)
    return P_w

# define fitting function via Eq. 2.100 and the Figure 2.11 in page 55########
def fit(freq,R,C,L):  
    lor = 1/(np.sqrt( 1+(1/(R*R))*((1/(2*np.pi*freq*C)-L*2*np.pi*freq)**2) ))

    return lor


fig, ax=plt.subplots()
freq=[]
P_dbm=[]
x=[]
y=[]

#### Read the csv data#######
f=pd.read_csv("RLC_trac_gen.csv", delimiter=";",header=0)


### Cut the range of the data where lorentzian distribution is relevant#########
newf=f[80:350]

###Continue reading the columns ########

freql=newf["Name"].tolist()
P_dbml=newf["Sweep (T1)"].tolist()


##### Create float arrays for x as the frequency and y as the dBm. ######
for i in range(len(freql)):
    x.append(freql[i].replace(",","."))
    y.append(P_dbml[i].replace(",","."))
    freq.append(float(x[i]))
    P_dbm.append(float(y[i]))
   
P_W=np.zeros(len(P_dbm))


#### Convert dBm to Watt #########
for kk in range(len(P_dbm)):
    P_W[kk]=conv(P_dbm[kk])

#### normalize Watt ######## 
   
P_W=P_W/max(P_W)


#### Guess parameters as follows [Rl (Ohm), C (pF), L(nH)] #####

Guess=[100,2.0e-12,100e-9]

#### Call the fitting function from scipy library  ########

popt,pcov=scipy.optimize.curve_fit(fit,freq,P_W,p0 = Guess)

#### optimised parameters popt provide the best values for R,C and L if your fitting is valid #########

print(popt)

###Creating and array of frequency between 5 kHz to 750 MHz #########

array_freq=np.linspace(5000,750000000,10000)
last_fit=np.zeros(len(array_freq))

#### Send the guess parameters extracted from popt as R,C,L #########
last_fit=[fit(l,popt[0],popt[1],popt[2]) for l in array_freq]

### Plot the original data and fitted data (red curve) check if the fit is okay. If no, change guess values. #####
ax.set_ylim([0,1])
#ax.set_xlim([0.15e9,0.5e9])
ax.plot(freq,P_W)
ax.plot(array_freq, last_fit,"r")
ax.set_xlabel("Freq (Hz)", fontsize=15)
ax.set_ylabel("Power (W)", fontsize=15)


print("this is the new version")
### Save the figure. #######
#plt.savefig("Lorenz_fit.pdf")
plt.show()
