import numpy as np
import matplotlib.pyplot as plt 


B_upper = 1.7527
B_lower = 1.6324

m = -14.06940981
v00 = 19378.5025

v_PR = v00 + (B_upper+B_lower)*m + (B_upper - B_lower)*(m**2)
print(v_PR)


