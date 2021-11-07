import numpy as np
from scipy.stats import alpha
from matplotlib import pyplot as plt

K = 100
N = int(10e6)
# xx = np.random.rand(N, K).sum(1)
xx = alpha.rvs(10, size=N)
plt.figure()
plt.hist(xx, bins=1000)
plt.show()