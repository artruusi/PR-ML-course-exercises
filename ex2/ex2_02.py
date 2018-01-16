import numpy as np
from numpy.linalg import inv

x = np.load("x.npy")
y = np.load("y.npy")

x = np.array([7,9,2])
y = np.array([11.6, 14.8, 3.5])

A = np.vstack([x, np.ones(len(x))]).T

m,c = np.linalg.lstsq(A,y)[0]

b = np.array([m,c]).T
