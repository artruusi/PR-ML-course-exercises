import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize_data(X):
    norm_X = X
    for n in range(0,2):
        for x in range(0, X[n].size):
            norm_X[n][x] = (X[n][x] - np.mean(X[n]))/np.std(X[n])

    return norm_X

filename = 'locationData.csv'

npArray = np.loadtxt(filename, delimiter=' ')
npArray.shape = (600,3)

norm_npArray = normalize_data(npArray)

print(np.mean(norm_npArray[0], axis=0))
print(np.std(norm_npArray[0], axis=0))

print(npArray)

npArray.shape = (3,600)
fig = plt.figure()
ax = plt.subplot(2,1,1)
plt.plot(npArray[0], npArray[1])
ax = plt.subplot(2, 1, 2, projection= "3d")
plt.plot(npArray[0],npArray[1],npArray[2])
plt.show()



