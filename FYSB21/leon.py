import numpy as np
from matplotlib import pyplot as plt 
import matplotlib.animation as animation
import scipy as sp

# TASK 1
# Equation (4)
# def u(x,t, ylim = 1/2, alpha = 1):
#     term1 = sp.special.erf((ylim - x)/(np.sqrt(4*alpha*t)))
#     term2 = sp.special.erf((-ylim - x)/(np.sqrt(4*alpha*t)))
#     return (1/2)*(term1 - term2)

# # Set up figure. 
# xVals = np.linspace(-10, 10, 100)
# tVals = np.linspace(0.1, 60, 500) # Not including 0. 
# yVals = [u(x, tVals[0]) for x in xVals]

# fig, ax = plt.subplots()
# ax.set(xlabel = "x", ylabel = "u(x)")
# line = ax.plot(xVals, yVals, label = f"time t =")[0] # y(x)
# ax.legend()

# def update(frame):
#     t = tVals[frame]
#     yVals = [u(x, t) for x in xVals]
#     line.set_ydata(yVals)
#     line.set_label(f"time t = {round(t, 4)}")
#     ax.legend()
#     return line 

# ani = animation.FuncAnimation(fig = fig, func = update, frames = 400, interval = 10)
# plt.show()


# TASK 2 

r0x = 17.5
r0y = 37.5
rmax = 2.5
u0 = 100
alpha = 2
X, Y = np.meshgrid(np.linspace(17.5 - 20, 17.5 + 20, 100), np.linspace(37.5 - 20, 37.5 + 20, 100))

def u(x, y, t = 0.01):
    vec = np.array([x, y]) - np.array([r0x, r0y])
    r = np.linalg.norm(vec)
    term1 = sp.special.erf((rmax - r)/np.sqrt(4*alpha*t))    
    term2 = sp.special.erf((-rmax - r)/np.sqrt(4*alpha*t))
    return (1/2)*u0*(term1 - term2)

uvec = np.vectorize(u)
Z = uvec(X, Y)

tVals = np.linspace(0.01, 50, 500)
fig, ax = plt.subplots()
ax.set(xlabel = "x", ylabel = "y")
ax.set_title(f"u(x,y) at time t = {round(tVals[0], 4)} units")
colormesh = ax.pcolormesh(X, Y, Z)
fig.colorbar(colormesh)


def update(framenum):
    time = tVals[framenum]
    Z = uvec(X, Y, t = time)
    colormesh.set_array(Z)
    ax.set_title(f"u(x,y) at time t = {round(time, 3)} units")
    return colormesh

ani = animation.FuncAnimation(fig = fig, func = update, frames = 200, interval = 1)
writergif = animation.PillowWriter(fps = 30)
ani.save("animation1.gif", writer = writergif)


    
    





    
    