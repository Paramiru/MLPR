import numpy as np
import matplotlib.pyplot as plt

def rbf_tut1_q3(xx, kk, hh):
    """Evaluate RBF kk with bandwidth hh on points xx (shape N,)"""
    ck = (kk - 51) * hh / np.sqrt(2)
    print(ck)
    return np.exp(-(xx - ck)**2 / hh**2)  # shape (N,)

def plot_rbf(xx, hh):
    xx = np.array(xx)
    plt.clf()
    for kk in range(1, 102):
        yy = rbf_tut1_q3(xx, kk, hh)
        plt.plot(xx, yy)
    plt.show()


# plotting code
N = 70
hh = 0.1
step_size = 2 / N
xx = np.arange(-1,1 + step_size, step_size)
print(xx)
plot_rbf(xx, hh)