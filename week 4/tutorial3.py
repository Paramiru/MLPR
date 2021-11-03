import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([0.5, 0.1, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.35, 0.25])[:,None]
x2 = np.array([0.9, 0.8, 0.75, 1.0])[:,None]
N = len(x1) + len(x2)
prior1 = len(x1) / N
prior2 = len(x2) / N

def gaussian(mean, var, xx):
    return np.exp(-0.5 * (xx - mean)**2 / var) * 1/np.sqrt(2*np.pi*var)

def plot_tutorial_3a():
    plt.clf()

    """Code for plotting the fitted Gaussians"""
    # X_grid = np.arange(-.1, 1.2, 0.01)[:,None]
    # gaussian_1 = gaussian(x1.mean(), x1.var(), X_grid)
    # gaussian_2 = gaussian(x2.mean(), x2.var(), X_grid)
    # plt.plot(X_grid, gaussian_1, label='Gaussian 1')
    # plt.plot(X_grid, gaussian_2, label='Gaussian 2')

    """ Code to plot the scores p(x,y) = P(y)p(x|y) of the given datapoints """
    xx = np.concatenate((x1,x2))
    p_x_given_class_1 = prior1 * gaussian(x1.mean(), x1.var(), xx)
    plt.scatter(xx, p_x_given_class_1, label='score for data given class 1')
    p_x_given_class_2 = prior2 * gaussian(x2.mean(), x2.var(), xx)
    plt.scatter(xx, p_x_given_class_2, label='score for data given class 2')

    """ Code to plot the scores p(x,y) = P(y)p(x|y) as a function of x"""
    xx = np.linspace(0, 1, num=100)
    p_x_given_class_1 = prior1 * gaussian(x1.mean(), x1.var(), xx)
    plt.plot(xx, p_x_given_class_1, label='score for class 1')
    p_x_given_class_2 = prior2 * gaussian(x2.mean(), x2.var(), xx)
    plt.plot(xx, p_x_given_class_2, label='score for class 2')

    # change labels and add legend
    plt.xlabel('location x')
    plt.ylabel('p(x,y)')
    plt.legend()

    # Show the plot
    plt.show()

def plot_gaussians_only():
    """Code for plotting only the Gaussians"""
    X_grid = np.arange(-.1, 1.2, 0.01)[:,None]
    gaussian_1 = gaussian(x1.mean(), x1.var(), X_grid)
    gaussian_2 = gaussian(x2.mean(), x2.var(), X_grid)

    plt.plot(X_grid, gaussian_1, label='Gaussian 1')
    plt.scatter(x1, np.zeros_like(x1), label='points x1')
    plt.plot(X_grid, gaussian_2, label='Gaussian 2')
    plt.scatter(x2, np.zeros_like(x2), label='points x2')
    plt.legend()
    plt.show()

def plot_probabilities_only(x):
    """Code for plotting only the scores p(x|y)"""
    p_x_given_class_1 = prior1 * gaussian(x1.mean(), x1.var(), x)
    plt.scatter(x, p_x_given_class_1, label='class 1')
    p_x_given_class_2 = prior2 * gaussian(x2.mean(), x2.var(), x)
    plt.scatter(x, p_x_given_class_2, label='class 2')
    plt.legend()
    plt.show()

def probability_of_class_1_given_x(x: float) -> float:
    # P(x=0.6) = P(x=0.6 | y=1)P(y=1) + P(x=0.6 | y=2)P(y=2)
    probability_of_x = gaussian(x1.mean(), x1.var(), x) * prior1 + gaussian(x2.mean(), x2.var(), x) * prior2

    # P(y=1 | x=0.6) = P(x=0.6 | y=1)P(y=1) / P(x=0.6)
    probability_asked = gaussian(x1.mean(), x1.var(), x) * prior1 / probability_of_x
    print(f'P(y=1|x={x}) = {probability_asked}')

probability_of_class_1_given_x(0.6)
plot_tutorial_3a()