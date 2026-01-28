# Basics 6: Linear Fit with NumPy
# Write a program that reads two columns of numbers from a file (basics6.txt) and calculates 
# a linear fit of the data. Remember to skip the header of the input file.

import numpy as np

def linear_fit(filename):
    
    with open(filename, "r") as data:
        x = []
        y = []
        for line in data:
            val = line.split() # space separated
            x.append(float(val[0]))
            y.append(float(val[1]))
        print(x)
        print(y)
    return None, None

if __name__ == "__main__":
    slope, intercept = linear_fit("./basics/basics6.txt")
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")