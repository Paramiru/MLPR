import numpy as np
import matplotlib.pyplot as plt

# Set up and plot the dataset
yy = np.array([1.1, 2.3, 2.9, -3.4]) # N,
X = np.array([[0.8], [1.9], [3.1], [-1.3]]) # N,1
plt.clf()
plt.plot(X, yy, 'x', markersize=15, mew=2)

# phi-functions to create various matrices of new features
# from an original matrix of 1D inputs.
def phi_linear(Xin):
    return np.hstack([np.ones((Xin.shape[0],1)), Xin])
def phi_quadratic(Xin):
    return np.hstack([np.ones((Xin.shape[0],1)), Xin, Xin**2])
def phi_general_polynomial(Xin, degree=5):
    return Xin ** np.arange(degree+1)
def fw_rbf(xx, cc):
    """fixed-width RBF in 1d"""
    return np.exp(-(xx-cc)**2 / 2.0)
def phi_rbf(Xin):
    return np.hstack([fw_rbf(Xin, -1.5), fw_rbf(Xin, 1), fw_rbf(Xin, 2.5)])    

def fit_and_plot(phi_fn, X, yy):
    # phi_fn takes N, inputs and returns N,D basis function values
    w_fit = np.linalg.lstsq(phi_fn(X), yy, rcond=None)[0] # D,
    X_grid = np.arange(-1.4, 3.3, 0.01)[:,None] # N,1
    f_grid = np.dot(phi_fn(X_grid), w_fit)
    plt.plot(X_grid, f_grid, linewidth=2)

fit_and_plot(phi_linear, X, yy)
fit_and_plot(phi_quadratic, X, yy)
fit_and_plot(phi_rbf, X, yy)
fit_and_plot(phi_general_polynomial, X, yy)
plt.legend(('data', 'linear', 'quadatic fit', 'rbf fit', '5th order polynomial'))
plt.xlabel('x')
plt.ylabel('f')

plt.show()