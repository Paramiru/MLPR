import numpy as np

D = 5
K = 3

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

x = np.random.randn(D, 1)
c = np.random.randn(K, 1)
v = np.random.randn(K, 1)
W = np.random.randn(K, D)
b = np.random.randn(1, 1)
print(v.shape)
a = np.matmul(W, x) + c
h = sigmoid(a)
print(h.shape)
f = np.matmul(v.T, h) + b


# Create function for error and expensively approximate gradients by finite
# differences:
E_fn = lambda W, x, c, b, v: np.matmul(v.T, sigmoid(np.matmul(W, x) + c)) + b
# E_fn = lambda x, A: np.matmul(x.T, np.matmul(A, x))
def checkgrad(fn, hh, *args):
    """Return all approx partial derivatives of fn wrt args"""
    bars = []
    for arg in args:
        bar = np.zeros_like(arg)
        arg_view = arg.ravel()
        bar_view = bar.ravel()
        for ii in range(arg_view.size):
            cc = arg_view[ii]
            arg_view[ii] = cc + hh/2.0
            f2 = fn(*args)
            arg_view[ii] = cc - hh/2.0
            f1 = fn(*args)
            arg_view[ii] = cc
            bar_view[ii] = (f2 - f1) / hh
        bars.append(bar)
    return bars
W_bar_fd, x_bar_fd, c_bar_fd, b_bar_fd, v_bar_fd = checkgrad(E_fn, 1e-5, W, x, c, b, v)

f_bar = 1
h_bar = v
a_bar = (1-sigmoid(a))*sigmoid(a)*h_bar
x_bar = np.matmul(W.T, a_bar)

err1 = np.max(np.abs(x_bar - x_bar_fd))
print(err1)
