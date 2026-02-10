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


