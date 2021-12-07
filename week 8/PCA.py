from matplotlib.colors import LightSource
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

N = 100
D = 2
sigma_a = 5
mu_a = 10
A = np.random.randn(N, D) * sigma_a + mu_a
# Get random covariance matrix 
# use A.T due to numpy's implementation
cov_A = np.cov(A.T)
print(cov_A)

# Obtain eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_A)
eigenvalues_matrix = eigenvalues[:, None].round(5) * np.eye(D)
print(eigenvalues_matrix)

samples_1 = np.random.randn(N, D)

# Scatter plot of data points
plt.figure(1, figsize=(8,8))
plt.scatter(samples_1[:,0], samples_1[:,1], color='g', s=3)
plt.xlim(-15,15)
plt.ylim(-15,15)
# Scatter plot after transforming with the eigenvalues
plt.figure(2, figsize=(8,8))
samples_2 = samples_1 @ np.sqrt(eigenvalues_matrix)
plt.scatter(samples_2[:, 0], samples_2[:, 1] , color='y', s=3)
plt.xlim(-15,15)
plt.ylim(-15,15)
# Scatter plot after performing rotation
plt.figure(3, figsize=(8,8))
L = eigenvectors @ np.sqrt(eigenvalues_matrix)
mu_v = np.array([mu_a]*2)
print(L @ L.T)
samples_3 = samples_2 @ L + mu_v
plt.scatter(samples_3[:,0], samples_3[:, 1], color='b', s=3)
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.show()