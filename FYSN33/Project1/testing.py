import numpy as np
import matplotlib.pyplot as plt
import time
import main
from matplotlib.pyplot import imshow


# # Broadcasting
# x = np.linspace(7, 45, 100)
# y = np.linspace(90, 5, 100)

# coly = y[:,np.newaxis] # shape as column vector

# # Broadcasting sum
# t0 = time.time()
# summ = x[:,np.newaxis] + y
# t1 = time.time()

# print("this took",  round(t1 - t0, 4), "s")

# # Double loop for matrix sum
# t0 = time.time()
# sumxy = np.zeros((len(x), len(y)))
# for i in range(len(x)):
#     for j in range(len(y)):
#         sumxy[i][j] = x[i] + y[j]        
# t1 = time.time()

# print("looping sum takes ", round(t1 - t0, 4), "seconds")




dim = 3
N = 3
# slicing matrix
S = np.zeros((N, 2*dim +2))

#print(S)

x = np.array([1, 100, 200])
y = np.array([1,2,3])
z = np.ones(N)

r = np.column_stack((x,y,z))
S[:,:dim] = r


print(r.shape)
ri = r[:,None,:]
#print(ri)
rj = r[None,:,:]

print(ri)
print()
print(rj)

print()
print(ri.shape)
print(rj.shape)

print(ri-rj)


# plt.imshow(S)
# plt.show()




# 3D table
# vx = np.array([1,2,3])
# vy = np.array([5,4,3])
# vz = np.array([0,0,0])

# v_vec = np.column_stack((vx,vy,vz))
# sys.S[:,dim:2*dim] = v_vec


# print("vx particle 0 = ", sys.v[0][0])
# print("vx particle 1 = ", sys.v[1][0])
# print("vx particle 2 = ", sys.v[2][0])

# print("vy particle 0 = ", sys.v[0][1])

# v[i][j][0] v_xi - v_xj
# v[i][j][1] v_yi - v_yj
# vij = sys.v[:,np.newaxis,:] - sys.v[np.newaxis,:,:]


# gradW[i, j, k]
# i = particle being updated
# j = neighbor particle
# k = x,y,z component




