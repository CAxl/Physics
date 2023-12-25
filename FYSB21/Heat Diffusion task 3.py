import numpy as np 
import scipy as sci 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import matplotlib.ticker as ticker

#------------------------
# Earth
# fixed layer radii
earthradius = 6371 * 1000   
earthcenter = np.array([0,0])   
r1 = (6371 - 5150)*1000     # inner core
r2 = (6371 - 2900)*1000     # core
r3 = (6371 - 100)*1000      # mantle
r4 = (6371) * 1000          # crust

# fixed earth temperatures
t1 = 6000   # inner core
t2 = 4000
t3 = 2500  
t4 = 500

# fixed earth diffusivity
a1 = 10**(-5)   # inner core
a2 = 10**(-6)
a3 = 10**(-7)
a4 = 10**(-8)

#----------------------------------
# meteor

u0 = 2.5*10**(5)  # initial impact temperature
r0 = np.array([(1/np.sqrt(2)) * earthradius, (1/np.sqrt(2)) * earthradius]) # impact point
rmax = 1.5*10**(5)     # radius of meteor

#----------------------------------
# coordinate system

xvals = np.linspace(0, 7 * 10**(6), 200)
yvals = np.linspace(0, 7 * 10**(6), 200)
xx,yy = np.meshgrid(xvals, yvals)

#----------------------------------
# fix initial temperatures

def layers(x,y):    # set base temps for layer radii
    vec = np.array([x,y])
    r = np.linalg.norm(vec)

    if 0 < r < r1:
        return t1, a1
    elif r1 < r <r2:
        return t2, a2
    elif r2 < r < r3:
        return t3, a3
    elif r3 < r < r4:
        return t4, a4   
    else:
        return 0, 0

basetemp = np.vectorize(layers)

#--------------------------------
# heat function

def u(x,y, t = 1):
    beta = 5*10**(-19)
    T, a = basetemp(x, y)
    if a == 0:
        return 0    # we're in space dude
    vec = np.array([x,y]) - r0
    r = np.linalg.norm(vec)
    term1 = sci.special.erf((rmax - r) / np.sqrt(4*a*t))
    term2 = sci.special.erf((-rmax - r) / np.sqrt(4*a*t))
    return T + ((1/2)*u0*(term1 - term2) * np.exp(-beta * t))

uvec = np.vectorize(u)
z = uvec(xx,yy)

#----------------------------------------------
# plotting

tvals = np.linspace(1, 3.154*10**(7)*10**(12), 1000)    # time count up to 10^12 years
fig, ax = plt.subplots()
colormesh = ax.pcolormesh(xx,yy, z, cmap=plt.cm.jet, vmin=0, vmax=10000)
ax.set_aspect("equal")
bar = fig.colorbar(colormesh)
bar.set_label("Temperature [K]")
scale_x = 1.e3  # scale to km
scale_y = 1.e3
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
ax.xaxis.set_major_formatter(ticks_x)
ax.yaxis.set_major_formatter(ticks_y)

#-----------------------------------------
# udate frames function

def update(framenum):  # things that update for every frame
    time = tvals[framenum]
    z = uvec(xx,yy, t = time)
    colormesh.set_array(z) 
    ax.set_title(f"u(x,y) at time t = {round(time*(3.154*10**(-8) * 10**(-9)), 1)} Giga years") # convert to years, then years to billion years
    ax.set(xlabel="x [km]", ylabel="y [km]")
    return colormesh

ani = animation.FuncAnimation(fig = fig, func = update, frames = 200, interval = 10)


#plt.show()
writergif = animation.PillowWriter(fps = 30)
ani.save("animation3.gif", writer = writergif)


