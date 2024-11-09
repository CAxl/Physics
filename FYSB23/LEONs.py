import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation 

# Globals. 
N = 32
a = 0.25
DELTATIME = np.sqrt(1/8)
TIMESTEPS = 40000

"""
Creates the A matrix and returns its eigenvalues and eigenvectors.
Return: tuple of arrays, ((N-1) eigenvalues, (N-1) x (N-1) eigenvectors)
"""
def get_A_evals_evecs():
    A = np.diag(np.full(N-1, 2.)) + np.diag(-np.ones(N-2), 1) + np.diag(-np.ones(N-2), -1)
    return np.linalg.eig(A)

"""
Orders the eigenvalues (lowest to highest) and corresponding vectors. 
@param evals: (N-1) array, the eigenvalues of A.
@param evecs: (N-1) x (N-1) array, the eigenvectors of A.
Return: tuple of arrays, ((N-1) w_n, (N-1) x (N-1) w^n). 
"""
def order_values(evals, evecs):
    nested = []
    ordered_vecs = np.full(np.shape(evecs), 0.)

    i = 0
    for val in evals:
        nested.append([val, i])
        i += 1
    
    nested.sort(key = lambda x: x[0])

    i = 0
    for nest in nested:
        ordered_vecs[:, i] = evecs[:, nest[1]]
        i += 1

    return np.sqrt(np.array([nest[0] for nest in nested])), ordered_vecs

"""
Determines the forces at each timestep. 
@param u: (N-1) array, the displacement array. 
Return: (N-1) array, the forces. 
"""
def get_Forces(u):
    Forces = []
    Forces.append(u[1] - 2*u[0] + a*(u[1] - u[0])**2 - a*u[0]**2)
    for i in range(1, N-2):
        Forces.append(u[i+1] - 2*u[i] + u[i-1] + a*(u[i+1] - u[i])**2 - a*(u[i] - u[i-1])**2)
    Forces.append(-2*u[-1] + u[-2] + a*u[-1]**2 - a*(u[-1] - u[-2])**2)
    return Forces

"""
Determines the energies of each normal mode, at each timestep. 
@param u: (N-1) array, the displacement array. 
@param v: (N-1) array, the velocity array. 
@param evals: (N-1) array, the w_n. 
@param evecs: (N-1) x (N-1) array, the w^n. 
Return: the array [E1, E2, E3, E4] evaluated at the current timestep. 
"""
def get_Energies(u, v, evals, evecs):
    xis = []
    dotxis = []
    Ens = []
    for n in range(4):
        xi = np.dot(evecs[:, n], u)
        xidot = np.dot(evecs[:, n], v)
        Ens.append((1/2)*(xidot**2 + (evals[n]**2)*(xi**2)))
    return Ens 


# displacements = []

def main():
    
    # Setup. 
    Energies = []
    Avals, Avecs = get_A_evals_evecs()
    eigenvalues, eigenvectors = order_values(Avals, Avecs)

    # Initialize displacement and velocity arrays. 
    uarr = [4*eigenvectors[:,0][i] for i in range(N-1)]
    varr = [0 for i in range(N-1)]
    
    # Timeloop. 
    for m in range(TIMESTEPS):
        Forces = get_Forces(uarr)
        new_uarr = [uarr[i] + varr[i]*DELTATIME + (1/2)*Forces[i]*DELTATIME**2 for i in range(N-1)]
        new_Forces = get_Forces(new_uarr)
        new_varr = [varr[i] + (1/2)*DELTATIME*(new_Forces[i] + Forces[i]) for i in range(N-1)]
        
        # Record the energies. 
        Energies.append(get_Energies(uarr, varr, eigenvalues, eigenvectors))

        # displacements.append(uarr)

        # Setup for new loop. 
        uarr = new_uarr
        varr = new_varr 
    
    # Produce plot. 
    E1 = [100*energies[0] for energies in Energies]
    E2 = [100*energies[1] for energies in Energies]
    E3 = [100*energies[2] for energies in Energies]
    E4 = [100*energies[3] for energies in Energies]

    time_axis = [m*DELTATIME*eigenvalues[0]/(2*np.pi) for m in range(TIMESTEPS)]

    plt.plot(time_axis, E1, label = "E1")
    plt.plot(time_axis, E2, label = "E2")
    plt.plot(time_axis, E3, label = "E3")
    plt.plot(time_axis, E4, label = "E4")
    plt.xlim(0, 160)
    plt.legend()
        
    return plt.show()

main()


# # Example data (replace this with your actual data)
# particle_positions_y = displacements

# # Number of particles
# num_particles = len(particle_positions_y[0])

# # Initialize the figure and axis
# fig, ax = plt.subplots()
# scatter = ax.scatter(range(num_particles), particle_positions_y[0]) # Scatter plot

# ax.set_ylim(-1, 1)

# def update(frame):
# scatter.set_offsets(list(zip(range(num_particles), particle_positions_y[frame])))
# return scatter,

# # Set the number of frames (timesteps)
# num_frames = len(particle_positions_y)

# # Create the animation
# ani = FuncAnimation(fig, update, frames=num_frames, blit=True)

# ani.save('particle_animation.gif', writer='imagemagick')

# # Show the animation
# plt.show()