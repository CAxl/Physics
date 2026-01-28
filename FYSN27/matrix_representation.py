import numpy as np


A = np.array([[4, 1, 1],
             [1, 3, 0],
             [1, 0, 2]])


eigvals = np.linalg.eigh(A)[0]
eigvecs = np.linalg.eigh(A)[1]

eigvecs = eigvecs.T
v1 = eigvecs[0]
v2 = eigvecs[1]
v3 = eigvecs[2]

print(v1)


print("Eigenvalues of A:")
print(eigvals)
print()
print("Eigenvectors of A (col major):")
print(eigvecs)
print()

V = np.array([v1, v2, v3])
print(V.T@A@V)  



 
def spectral(lamb, vecs):

    N = len(vecs)
    V = np.zeros((N,N))
    
    tol = 0.0001
    
    for i in range(N):
        for j in range(N):
            
            V[i][j] += lamb[i] * vecs[i].T@vecs[j]
            
            if V[i][j] < tol: 
                V[i][j] = 0
            
    return V

M = spectral(eigvals, V)
print(M)


def spectral(lamb, vecs):
    N = len(lamb)
    A_reconstructed = np.zeros((N, N))
    for i in range(N):
        v = vecs[i][:, np.newaxis]  # shape (N,1)
        A_reconstructed += lamb[i] * (v @ v.T)  # outer product
    return A_reconstructed



eigvals, eigvecs = np.linalg.eigh(A)
# keep eigenvectors as columns
vecs = [eigvecs[:, i] for i in range(len(eigvals))]
M = spectral(eigvals, vecs)
print(M)  # should match A




