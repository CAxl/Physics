import numpy as np
import matplotlib.pyplot as plt 

plt.rcParams['text.usetex'] = True

# TASK 1
N = [20, 40, 80, 160]
r_ee_squared = [66.7, 193, 549, 1555]
logx = np.log(N)
logy = np.log(r_ee_squared)

Sfit = np.poly1d(np.polyfit(logx,logy,1))   # fitting self avoiding walk
Sline = np.linspace(3,5.5,100)

Ofit = np.poly1d(np.polyfit(logx,logx,1))   # fitting ordinary walk
Oline = np.linspace(3,5.5, 100)

plt.plot(Sline, Sfit(Sline), linestyle="--", label=r"$k = 2\nu$")
plt.plot(Oline, Ofit(Oline),'orange', linestyle="--")
plt.scatter(logx, logy, label="Self avoiding random walk")
plt.scatter(logx, logx, label="Ordinary random walk")
plt.xlabel(r"$\log(N)$")
plt.ylabel(r"$\log(r_{ee}^2)$")
plt.grid()
plt.legend()
plt.show()


# TASK 2

eps = k = 1
E_min = -13*eps

T = np.linspace(0.1, 1, 100)
E_eps = np.linspace(0,13,14)*(-1)
print(E_eps)
g_E = np.array([18671059783.5, 15687265041, 5351538782, 1222946058, 234326487, 
                40339545, 5824861, 710407, 77535, 9046, 645, 86, 0, 1])

def expectedE_sq(T):
    num = 0
    denom = 0
    i=0
    for Eval in E_eps:
        num += Eval*g_E[i]*np.exp(-(Eval-E_min)/(k*T))
        denom += g_E[i]*np.exp(-(Eval-E_min)/(k*T))
        i+=1
    return (num/denom)**2

def expected_Esq(T):
    num = 0
    denom = 0
    i=0
    for Eval in E_eps:
        num += pow(Eval,2)*g_E[i]*np.exp(-(Eval-E_min)/(k*T))
        denom += g_E[i]*np.exp(-(Eval-E_min)/(k*T))
        i+=1
    return (num/denom)

def expected_Cv_T(T,E1,E2):
    return (1/(k*T**2))*(E1-E2)

Cv = expected_Cv_T(T,expected_Esq(T), expectedE_sq(T))
T_max =(T[np.where(Cv==max(Cv))]) # prints max(Cv)
plt.axvline(T[np.where(Cv==max(Cv))], linestyle="--", color='orange')
Cvmax = Cv[np.where(Cv==max(Cv))]
plt.plot(T_max, Cv[np.where(Cv==max(Cv))], "o",color='orange', label=f"T = {np.round(float(T_max),2)}")
plt.plot(T,Cv)
plt.xlabel(r"T [$\epsilon/k_B$]")
plt.ylabel(r"Cv [$k_B$]")
plt.grid()
plt.legend()
plt.show()


# task 2c

def Pnat(T):
    i = 0
    denom = 0
    for Eval in E_eps:
        denom += g_E[i]*np.exp(-(Eval)/(k*T))
        i += 1
    return np.exp(-E_min/(k*T)) / denom

plt.plot(T, Pnat(T))
plt.xlabel(r"T [$\epsilon/k_B$]")
#plt.axvline(T[np.where(Cv==max(Cv))], linestyle="--", color='orange')
plt.ylabel(r"P$_{nat}$(T)")
plt.grid()
plt.show()





