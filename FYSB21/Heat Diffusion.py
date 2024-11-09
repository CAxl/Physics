import numpy as np
import scipy as sci 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation


a = 1  # diffusivity
a2 = 10
r1 = 5

y_pos = 0.5
y_neg = -y_pos

xvals = np.linspace(-20, 20, 1000)

#-------------
# diffusivity as fnc of position

def diffuse(x):
    vec = np.array([x])
    r = np.linalg.norm(vec)
    if 0 < r < r1:
        return a
    else:
        return a2

#-----------------------
# 1-D heat func

def u(x,t):     
    a = diffuse(x)
    return 1/2 * (sci.special.erf((y_pos - x)/np.sqrt(4*a*t)) - 
                  sci.special.erf((y_neg - x)/np.sqrt(4*a*t)))


t = np.linspace(0.01, 10, 500)
yvals = [u(x, t[0]) for x in xvals]     # y = u(x,t)

# ---------------
# plotting

fig, ax = plt.subplots()
line = ax.plot(xvals,yvals)[0]    # graph of u(x,t)
ax.set_ylabel("u(x)")
ax.set_xlabel("x")
ax.tick_params(direction="in",top="True",right="True")


def update(frame):  # things that update every frame
    time = t[frame]
    yvals = [u(x,time) for x in xvals]
    line.set_ydata(yvals)
    ax.set_title(f"u(x) at time t = {round(time,1)} [unit time]")
    return line

ani = animation.FuncAnimation(fig = fig, func = update, frames = 200, interval = 1)

plt.show()
# writergif = animation.PillowWriter(fps = 30)
# ani.save("animation1.gif", writer = writergif)
