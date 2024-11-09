from numpy import *

def P1s(r,Z):
    return 2*Z**(3/2)*r*exp(-Z*r)

def P2s(r,Z):
    return (1/sqrt(2))*Z**(3/2)*r*exp(-Z*r/2)*(1-(1/2)*Z*r)

def P2p(r,Z):
    return (1/(2*sqrt(6)))*Z**(5/2)*r**2*exp(-Z*r/2)

def P3s(r,Z):
    return (2/(3*sqrt(3)))*Z**(3/2)*r*exp(-Z*r/3)*(1-(2/3)*Z*r+(2/27)*Z**2*r**2)

def P3p(r,Z):
    return (8/(27*sqrt(6)))*Z**(5/2)*r**2*exp(-Z*r/3)*(1-(1/6)*Z*r)

def P3d(r,Z):
    return (4/(81*sqrt(30)))*Z**(7/2)*r**3*exp(-Z*r/3)


def P_function(r,Z,n,l):
    if   n == 1 and l == 0:
        return 2*Z**(3/2)*r*exp(-Z*r)
    
    elif n == 2 and l == 0:
        return (1/sqrt(2))*Z**(3/2)*r*exp(-Z*r/2)*(1-(1/2)*Z*r)
    
    elif n == 2 and l == 1:
        return (1/(2*sqrt(6)))*Z**(5/2)*r**2*exp(-Z*r/2)
    
    elif n == 3 and l == 0:
        return (2/(3*sqrt(3)))*Z**(3/2)*r*exp(-Z*r/3)*(1-(2/3)*Z*r+(2/27)*Z**2*r**2)
    
    elif n == 3 and l == 1:
        return (8/(27*sqrt(6)))*Z**(5/2)*r**2*exp(-Z*r/3)*(1-(1/6)*Z*r)
    
    elif n == 3 and l == 2:
        return (4/(81*sqrt(30)))*Z**(7/2)*r**3*exp(-Z*r/3)
    
