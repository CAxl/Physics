import numpy as np



p1 = np.array([-16.481, 277.490])
p2 = np.array([-19.906, 303.924])

q1 = np.array([130.034, 404.254])
q2 = np.array([128.036, 415.563])


# Direction vectors
dP = p2 - p1
dQ = q2 - q1

# Set up system: P1 + t*dP = Q1 + s*dQ  =>  t*dP - s*dQ = Q1 - P1
A = np.column_stack((dP, -dQ))
b = q1 - p1

# Solve for t and s
try:
    t_s = np.linalg.solve(A, b)
    intersection = p1 + t_s[0] * dP
    print("Intersection point:", intersection)
except np.linalg.LinAlgError:
    print("Lines are parallel or coincident; no unique intersection.")