import numpy as np 
import matplotlib.pyplot as plt 

grid_size = 0.01
K = 5
ww = np.random.randint(0,10, size=K) / 3

plt.clf()

def general_polynomial(Xin, degree):
    Xin = np.array(Xin)
    return Xin ** np.arange(degree+1)

def quadratic(Xin):
    return np.hstack([np.ones((Xin.shape[0], 1)), Xin, Xin**2])

def fw_rbf(xx, cc):
    """fixed-width RBF in 1d"""
    return np.exp(-(xx-cc)**2 / 2.0)

def phi_rbf(Xin):
    return np.hstack([fw_rbf(Xin, 1), fw_rbf(Xin, 2), fw_rbf(Xin, 3)])

X_grid = np.arange(-3, 3, grid_size)[:, None] # N,1 

# plt.plot(X_grid, phi_rbf(X_grid) @ ww, linewidth=2)
# plt.plot(X_grid, quadratic(X_grid) @ ww, linewidth=2)
plt.plot(X_grid, general_polynomial(X_grid, 4) @ ww, linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(('RBF', 'Quadratic'))
plt.show()

print(phi_rbf(X_grid))