import numpy as np

n = '\n'

def check_covariance_2(x, N) -> np.array:
    x = np.array(x)
    means = x.mean(axis=0)[:, None]
    covariance_matrix = (x.T @ x) / N - means @ means.T
    print(means)
    print(means @ means.T)
    print(f'means array: {n} {means} {n}')
    print(f'means array transposed: {n} {means.T} {n} ')
    print(f'Covariance matrix: {n} {covariance_matrix}  {n}')
    
x1 = np.random.randn(10**6)
# compute covariance when variables are the same
# hint1: the covariance of the same vector is the variance
# hint2: variance of a normal distribution is 1
X = np.hstack((x1[:,None], x1[:,None])) 
# Y = np.random.randn(10**6, 2)
check_covariance_2(X, N=10**6)
print(f'Numpy\'s covariance matrix: {n} {np.cov(X.T)} {n}')

# print(np.isclose(1.00201491,1.00201592))