import numpy as np 
import matplotlib.pyplot as plt 
import sympy as sp
from matplotlib.animation import FuncAnimation


N = 32   # number of springs 
TIMESTEPS = 50000
a = 0.25
deltaTime = np.sqrt(1/8)

def get_eigA():
    A = np.diag(np.full(N-1,2.)) + np.diag(-np.ones(N-2), 1) + np.diag(-np.ones(N-2), -1)
    eigenvals = np.linalg.eig(A)[0]
    eigenvecs = np.linalg.eig(A)[1]
    return eigenvals, eigenvecs

print(type(get_eigA()[0][0]))

def sort_eig(e_vals, e_vecs):
    nested = []
    i = 0
    for val in e_vals:
        nested.append([val,i])  # save the corresponding index
        i+=1
    nested.sort(key = lambda x: x[0])   # sort eigenvalues from smallest to largest

    i = 0
    matrx = np.full(np.shape(e_vecs), 0.)
    for nest in nested:
        matrx[:,i] = e_vecs[:,nest[1]]    # list of eigvecs sorted in order of corresp eigvals
        i += 1

    return np.sqrt(np.array([nest[0] for nest in nested])), matrx


def get_f(u):
    forces = []
    forces.append(u[1] - 2*u[0] + a*(u[1] - u[0])**2 - a*u[0]**2)                           # force on first particle
    for i in range(1, N-2):                                                                             # second particle to second last particle (last: N-2, up to not including)
        forces.append(u[i+1] - 2*u[i] + u[i-1] + a*(u[i+1]- u[i])**2 - a*(u[i] - u[i-1])**2)
    forces.append(-2*u[-1] + u[-2] + a*u[-1]**2 - a*(u[-1] - u[-2])**2)                     # force on last particle
    return forces


def get_E(u,v,evals,evecs):
    """
    four eigemodes,
    define xi and time derivative of xi
    return E_n(xi, xidot)
    """
    En = []
    for n in range(0,4):
        xi = np.dot(evecs[:,n],u) # access n:th col in eigvec-matrix
        xi_dot = np.dot(evecs[:,n],v)
        En.append((1/2)*((xi_dot**2) + (evals[n]**2)*(xi**2)))
    return En 

displacements = []

def main():
    # setup
    Avals,Avecs = get_eigA()
    eigenvalues, eigenvectors = sort_eig(Avals, Avecs)

    energies = []

    # initial conditions
    u_arr = [4*eigenvectors[:,0][i] for i in range(N-1)]    
    v_arr = [0 for i in range(N-1)]
    
    # time loop
    for m in range(TIMESTEPS):
        forces = get_f(u_arr)
        
        new_u = [u_arr[i] + v_arr[i]*deltaTime + (1/2)*forces[i]*deltaTime**2 for i in range(N-1)]
        new_forces = get_f(new_u)
        new_v = [v_arr[i] + (1/2)*deltaTime*(new_forces[i] + forces[i]) for i in range(N-1)]

        energies.append(get_E(u_arr,v_arr,eigenvalues, eigenvectors))

        u_arr = new_u   # finish loop by reassignment
        v_arr = new_v
        displacements.append(u_arr)

    # energies to plot
    E1 = [100* energies[0] for energies in energies]
    E2 = [100* energies[1] for energies in energies]
    E3 = [100* energies[2] for energies in energies]
    E4 = [100* energies[3] for energies in energies]

    # time interval
    tvals = [m*deltaTime*eigenvalues[0] / (2*np.pi) for m in range(TIMESTEPS)]

    plt.plot(tvals, E1, label="E1")
    plt.plot(tvals, E2, label="E2")
    plt.plot(tvals, E3, label="E3")
    plt.plot(tvals, E4, label="E4")
    plt.xlim(0,160)
    plt.legend()
    plt.show()

main()


# Example data (replace this with your actual data)
particle_positions_y = displacements

# Number of particles
num_particles = len(particle_positions_y[0])

# Initialize the figure and axis
fig, ax = plt.subplots()
scatter = ax.scatter(range(num_particles), particle_positions_y[0]) # Scatter plot

ax.set_ylim(-1, 1)

def update(frame):
    scatter.set_offsets(list(zip(range(num_particles), particle_positions_y[frame])))
    return scatter,

# Set the number of frames (timesteps)
num_frames = len(particle_positions_y)

# Create the animation
ani = FuncAnimation(fig, update, frames=num_frames, blit=True)

ani.save('particle_animation.gif', writer='imagemagick')

# Show the animation
plt.show()




