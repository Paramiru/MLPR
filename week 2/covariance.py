import numpy as np
import matplotlib.pyplot as plt

def check_covariance_2(x) -> np.array:
    x = np.array(x)
    means = x.mean(axis=0)[:, None]
    covariance_matrix = (x.T @ x) / x.shape[0] - means @ means.T
    print(f'\nmeans array transposed:\n {means.T}')
    print(f'\nCovariance matrix:\n {covariance_matrix}')
    
N, K = 10**6, 2
xx = np.random.randn(N, K)
check_covariance_2(xx)
print(f'\nNumpy\'s covariance matrix:\n {np.cov(xx.T)}')

# compute covariance when variables are the same
# hint1: the covariance of the same vector is the variance
# hint2: variance of a normal distribution is 1
x1 = np.random.randn(10**6)
X = np.hstack((x1[:,None], x1[:,None])) 
check_covariance_2(X)
print(f'\nNumpy\'s covariance matrix:\n{np.cov(X.T)}')

# Probability density zero almost everywhere unless x1 = x2
# density along the line x1 = x2 is infinite
plt.scatter(X[:,0], X[:,1])
plt.show()
