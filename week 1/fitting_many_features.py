import numpy as np
import matplotlib.pyplot as plt

N,D = 10, 2
mu = np.random.rand(N)
X = np.tile(mu[:,None], (1, D)) + 0.01*np.random.randn(N, D)
yy = 0.1*np.random.randn(N) + mu

w_fit = np.linalg.lstsq(X, yy, rcond=0)[0]

grid_size = 0.01
X_grid = np.arange(-10, 10, grid_size)[:, None]

f_val = X @ w_fit

# plot will not work check 
# https://hyp.is/-6JNLh4MEeybMu8xJ8bzug/mlpr.inf.ed.ac.uk/2021/notes/w1d_linear_regression_regularization.html
# plt.clf()
# plt.plot(X, f_val)
# plt.show()
