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

# def sigmoid(X):
#     return 1/(1 + np.exp(-X))

# def neural_net(X):
#     """X is a (N,1) np.array"""
#     N = X.shape[0]
#     for _ in range(100):
#         W_1 = np.random.randn(N, N)
#         b_1 = 10
#         h_1 = sigmoid(W_1 @ X + b_1)
#     for _ in range(50):
#         W_2 = np.random.randn(N, N)
#         b_2 = 5
#         h_2 = sigmoid(W_2 @ h_1 + b_2)
#     W_f = np.random.randn(N, N)
#     b_f = 2
#     ff = W_f @ h_2 + b_f
#     return ff

def sigmoid(a): return 1. / (1. + np.exp(-a))
def relu(x): return np.maximum(x, 0)
def linear(a): return a

def neural_net(X, layer_sizes=(100,50,1), gg=linear, sigma_w=1):
    for out_size in layer_sizes:
        Wt = sigma_w * np.random.randn(X.shape[1], out_size)
        X = gg(X @ Wt)
        print(X.shape)
    return X

N = 100
X = np.linspace(-2, 2, num=N)[:, None]  # N,1
plt.clf()
for i in range(12):
    ff = neural_net(X)
    plt.plot(X, ff);

plt.show()