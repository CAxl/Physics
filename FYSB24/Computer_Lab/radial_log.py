# Program radiallog.py
# Program for solving the radial SchrÃ¶dinger equation for a Coulomb potential V(r)=-Z/r and a
# given quantum number l. A transformation to logaritmic grid is done to increase numerical
# accuracy. A specific solution is targeted by giving the principal quantum number n.
#                                                    JÃ¶rgen Ekman and Per JÃ¶nsson, April 2015 
# Matlab to python conversion by Asimina Papoulia and Wenxian Li, September 2020

import math 
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime

def radiallog(l,n,Z,plot=True):

    starttime=datetime.datetime.now().timestamp()
    
    if Z > 0 and l < n:
        E = - Z**2/2                                      #starting energy taken as the lowest possible
        nodes = n - l - 1                                 #targeted number of nodes
    else:
        sys.stderr.write('Invalid input\n')
        sys.exit(-1)
    
    rhomin = -10
    h = 1/48                                              #step size
    nodes_count = -1                                      #initial values of nodes
    dE = math.inf                                         #initial values of difference of derivat at r_c
    E_upper = -2.0e-52                                    #initial values of upper bound of energy during iter
    E_lower = -math.inf                                   #initial values of lower bound of energy during iter
    num_iter = 0
    
    while nodes != nodes_count or abs(dE) > 1.0e-10:
        #Make sure that the energy is such that the turning point is in the interior of the grid
        while 1:
            r_inf = 40/math.sqrt(2*abs(E))                              #practical infinity
            rhoi = math.log(Z*r_inf);                                   # practical infinity in rho
            grid_points = round((-rhomin + rhoi)/h);                    # number of grid points
            rho = rhomin + np.linspace(0,grid_points-1,grid_points)*h;  # generate grid
            r = np.exp(rho)/Z;
            # Define the effective potential 
            U = np.linspace(0, 0, grid_points)            #initialize U
            U[0] = 0                                      #gets a  0  value at r = 0
            U[1:] = -Z/r[1:] + l*(l+1)/(2*r[1:]**2)
    
            #Determine the outer classical turning point. Start from the practical infinity
            # and step inwards until U(i) < E
            i_c = grid_points-1
            for i in range(grid_points-1, -1, -1):
                if U[i] < E:
                    i_c = i
                    break
            #If turning point not in the interior increase the energy and try again
            if i_c == grid_points-1:
                E = 0.9*E
            else:
                break
    
        g = -2*r**2*(E + Z/r) + (l+1/2)**2;               #g function for Numerow
        alpha = 1-(h**2/12)*g                             #alpha and
        beta = 2+(5*h**2/6)*g                             #beta for Numerov's method
    
        #Perform the outward integration
        P_out = np.zeros(grid_points)                               #initialize P_out
        P_out[0] = r[0]**(l+1)*(1-Z*r[0]/(l+1))/math.sqrt(r[0])     #starting values from regular
        P_out[1] = r[1]**(l+1)*(1-Z*r[1]/(l+1))/math.sqrt(r[1])     #solution at r = 0
        for i in range(1,i_c+1):
            P_out[i+1] = (beta[i]*P_out[i] - alpha[i-1]*P_out[i-1])/alpha[i+1]
    
        #Perform the inward integration
        P_in = np.zeros(grid_points)                                                                    #initialize P_in
        P_in[grid_points-1] = np.exp(-np.sqrt(2*abs(E)*r[grid_points-1]))/math.sqrt(r[grid_points-1])   #starting values from
        P_in[grid_points-2] = np.exp(-np.sqrt(2*abs(E)*r[grid_points-2]))/math.sqrt(r[grid_points-2])   #regular solution at r_inf
        for i in range(grid_points-2,i_c-1,-1):
            P_in[i-1] = (beta[i]*P_in[i] - alpha[i+1]*P_in[i+1])/alpha[i-1]
    
        #Scale and merge solutions
        P_out = P_out/P_out[i_c]
        P_in = P_in/P_in[i_c]
        P = np.append(P_out[0:i_c], P_in[i_c:])
    
        #Count the number of nodes. At a node position, P(i+1)P(i) is negative
        nodes_count = 0
        for i in range(1,grid_points-1):
            if P[i+1]*P[i] < 0:
                nodes_count +=1
    
        #Check nodes and adjust E
        if nodes_count > nodes:                           #too many nodes, decrease E
            E_upper = min(E_upper,E)
            if E*1.1 < E_lower:
                E = (E_lower+E)/2
            else:
                E = E*1.1
        elif nodes_count < nodes:                         #too few nodes, increase E
            E_lower = max(E_lower,E)
            if E*0.9 > E_upper:
                E = (E+E_upper)/2
            else:
                E = E*0.9
    
        #Fine tune the energy
        if nodes_count == nodes:
            #Compute the derivatives and take difference
            DP_out = (P_out[i_c+1]-P_out[i_c-1])/(2*h)
            DP_in = (P_in[i_c+1]-P_in[i_c-1])/(2*h)
            dE = (DP_out-DP_in)*P[i_c]/(2*sum(r**2*P**2)*h)
            #Damp if step is too large
            while E+dE > 0:
                dE = dE/2
            E = E + dE
        print("Iteration %d: E = %.16e" %(num_iter,E,))
        num_iter += 1
    
    #Normalize and check slope
    norm = math.sqrt(sum(P**2*r**2)*h)
    P = P/norm
    if P[1] < 0:
        P = -P
    
    #Compute radial expectation value
    r1 = sum(r**3*P**2)*h
    
    endtime=datetime.datetime.now().timestamp()
    print()
    print("time used: %f s" %(endtime-starttime))
    
    #Write energy and plot solution
    print()
    print("Number of iterations: %d" %(num_iter,))
    print("Number of grid points: %d" %(grid_points,))
    print("Energy eigenvalue: %.16e a.u." %(E,))
    print("Radial expectation value: %.16e a.u." %(r1,))
    
    orbital=["s","p","d","f","g","h","i"]
    
    if plot:
        plt.figure()
        plt.subplot(211)
        plt.plot(r,np.sqrt(r)*P)
        plt.xlabel("r (a.u.)")
        plt.title("radial function for %d%s of Z = %d" %(n,orbital[l],Z,))
        plt.ylabel("P")
        rr = np.sqrt(r)
        plt.subplot(212)
        plt.plot(rr,np.sqrt(r)*P)
        plt.xlabel('$\sqrt{\mathrm{r}}$ (a.u.)')
        plt.ylabel("P")
        plt.tight_layout(pad=1.0)
        plt.show()

    return (r,P,E,)
 
if __name__ == '__main__':  
    #input
    l = int(input("Give the orbital quantum number l: "))
    n = int(input("Give the principal quantum number n: "))
    Z = int(input("Give the nuclear charge Z: "))
    print()
    radiallog(l,n,Z)
