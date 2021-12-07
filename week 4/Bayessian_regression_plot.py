import numpy as np
import matplotlib.pyplot as plt

def get_Bayessian_line():
    mu, sigma_slope, sigma_interc = 0, 0.4, 0.4
    slope_weight = np.random.normal(mu, sigma_slope)
    intercept_weight = np.random.normal(mu, sigma_interc)
    ww = np.array([[slope_weight, intercept_weight]]).T
    step_size = 0.01
    xx = np.arange(-5, 5, step=step_size)[:,None]
    ones = np.ones((xx.shape[0], 1))
    xx = np.concatenate((xx, ones), axis=1)
    return xx @ ww

def plot_Bayessian_lines(number_of_lines=50):
    plt.clf()
    step_size = 0.01
    X_grid = np.arange(-5, 5, step=step_size)[:, None]
    for i in range(number_of_lines):
        yy = get_Bayessian_line()
        plt.plot(X_grid, yy, label=f'line {i}')
    plt.show()

plot_Bayessian_lines()
    

