import numpy as np
import matplotlib.pyplot as plt

def get_Bayessian_line():
    mu, sigma = 0, 0.1
    slope_weight = np.random.normal(0, 0.04)
    intercept_weight = np.random.normal(0, 0.1)
    ww = np.array([[slope_weight, intercept_weight]]).T
    xx = np.arange(-5, 5, step=0.01)[:,None]
    ones = np.ones((xx.shape[0], 1))
    xx = np.concatenate((xx, ones), axis=1)
    return xx @ ww

def plot_Bayessian_lines(number_of_lines=50):
    plt.clf()
    X_grid = np.arange(-5, 5, step=0.01)[:, None]
    for i in range(number_of_lines):
        yy = get_Bayessian_line()
        plt.plot(X_grid, yy, label=f'line {i}')
    plt.show()

plot_Bayessian_lines()
    

