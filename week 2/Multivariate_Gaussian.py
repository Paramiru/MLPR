import numpy as np
import matplotlib.pyplot as plt

# Code to see how A transforms an independent
# 2D Gaussian distribution

# a = 1
# A = np.array([[1,0], [a, 1-a]])

A = np.array([[5,0], [0, 20]])
N,D = int(1e4), 2

x = np.random.randn(N,D)
z = A @ x.T
z = z.T

plt.clf()
plt.plot(z[:,0], z[:,1], '.')
plt.axis('square')
plt.show()