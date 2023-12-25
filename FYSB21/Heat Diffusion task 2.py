import numpy as np
import scipy as sci 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

# task 2: 2-D diffusion

a = 2
u0 = 100
r0x = 17.5
r0y = 37.5
rpos = 2.5
rneg = -rpos


xvals = np.linspace(r0x - 20, r0x + 20, 100)
yvals = np.linspace(r0y - 20, r0y + 20, 100)
X, Y = np.meshgrid(xvals, yvals)  

def u(x,y, t=0.01): # 2-D heat equation
    vec = np.array([x,y]) - np.array([r0x, r0y])
    r = np.linalg.norm(vec)
    return (1/2) * u0 * (sci.special.erf((rpos - r) / np.sqrt(4*a*t)) - 
                        sci.special.erf((rneg - r) / np.sqrt(4*a*t)))

uvec = np.vectorize(u)
Z = uvec(X, Y)

tvals = np.linspace(0.01, 50, 400)
fig, ax = plt.subplots()
colormesh = ax.pcolormesh(X,Y, Z, cmap=plt.cm.jet)
ax.set_aspect("equal")
bar = fig.colorbar(colormesh)
bar.set_label("Temperature [unit temperature]")

def update(framenum):  # things that update for every frame
    time = tvals[framenum]
    Z = uvec(X,Y, t = time)
    colormesh.set_array(Z)
    ax.set_title(f"u(x,y) at time t = {round(time, 1)} unit time")
    ax.set(xlabel="x", ylabel="y")
    return colormesh

ani = animation.FuncAnimation(fig = fig, func = update, frames = 200, interval = 10)

plt.show()
# writergif = animation.PillowWriter(fps = 30)
# ani.save("animation2.gif", writer = writergif)





