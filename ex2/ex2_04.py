import matplotlib.pyplot as plt
from scipy.io import loadmat

mat = loadmat("twoClassData.mat")

print(mat.keys())

X = mat["X"]
y = mat["y"].ravel()

plt.plot(X[y == 0, 0], X[y == 0, 1], 'ro')
plt.plot(X[y == 1, 0], X[y == 1, 1], 'bo')
