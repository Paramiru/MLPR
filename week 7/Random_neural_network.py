"""
Replace the neural_net function with one that evaluates
a random neural network function with two hidden layers 
with H1=100 and H2=50 hidden units, a scalar output, 
and logistic-sigmoid non-linearities. Sample all of the 
weights randomly from a standard normal. In the first instance,
you can omit the bias parameters. 
"""

import numpy as np
from matplotlib import pyplot as plt

def sigmoid(a): return 1. / (1. + np.exp(-a))
def relu(x): return np.maximum(x, 0)
def linear(a): return a

def neural_net(X, layer_sizes=(100,50, 1), gg=relu, sigma_w=1):
    for out_size in layer_sizes:
        bias = np.random.uniform(out_size)
        Wt = sigma_w * np.random.uniform(size=(X.shape[1], out_size))
        X = gg(X @ Wt)
        # print(X.shape + bias)
    return X

N = 100
X = np.linspace(-2, 2, num=N)[:, None]  # N,1
plt.clf()
for i in range(12):
    ff = neural_net(X, sigma_w=0.1)
    plt.plot(X, ff);

plt.show()