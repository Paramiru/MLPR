import numpy as np
import matplotlib.pyplot as plt

# set up and plot the dataset
yy = np.array([1.1, 2.3, 2.9]) # N,
X = np.array([[0.8], [1.9], [3.1]]) # N,1
plt.clf()
plt.plot(X, yy, 'x', markersize=20, mew=2)
K = 5

def fw_rbf(xx, cc=5, hh=0):
    return np.exp(-(xx-cc)**2 / hh)

def polynomial(xx, degree=K-1):
    return xx ** np.arange(degree+1)

def phi_rbf(Xin):
    return np.hstack([fw_rbf(Xin, -5, 1.0), fw_rbf(Xin, -2, 1.0), fw_rbf(Xin, 0, 1.0),
    fw_rbf(Xin, 1, 1.0), fw_rbf(Xin, 2, 1.0), fw_rbf(Xin, 4, 1.0), fw_rbf(Xin, -7, 1.0),
    fw_rbf(Xin, 7, 1.0), fw_rbf(Xin, -9, 1.0), fw_rbf(Xin, 9, 1.0), fw_rbf(Xin, 12, 1.0), 
    fw_rbf(Xin, 13, 1.0)])

grid_size = 0.01
X_grid = np.arange(0, 3.3, grid_size)[:, None] # N, 1

def fit_and_plot(phi_fn, X, yy, lmbda=0):
    y_tilde = np.hstack((yy, np.zeros(K)))
    phi_tilde = np.vstack((phi_fn(X), np.eye(K) * np.sqrt(lmbda)))
    w_fit = np.linalg.lstsq(phi_tilde, y_tilde, rcond=None)[0]
    f_val = phi_fn(X_grid) @ w_fit
    plt.plot(X_grid, f_val, linewidth=2)

# fit_and_plot(phi_rbf, X, yy)
fit_and_plot(polynomial, X, yy)
plt.legend(('data', '11th order polynomial'))
plt.xlabel('x')
plt.ylabel('y')
plt.show()