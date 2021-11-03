import numpy as np

N = 10 ** 6
# pick m, sigma, alpha and n
# estimate mean and covariance of the samples
m = 5
sigma = 3
alpha = 2
n = 1.5

def get_expected_mean() -> np.array:
    mean = [m, alpha * m]
    return np.array(mean)

def get_expected_covariance() -> np.array:
    variance_x1 = sigma ** 2
    variance_x2 = (alpha * sigma) ** 2 + n ** 2
    anti_diagonal_term = alpha * sigma ** 2
    covariance = [
        [variance_x1, anti_diagonal_term],
        [anti_diagonal_term, variance_x2]
    ]
    return np.array(covariance)

nu = np.random.normal(0, n, N)[:, None]
x1 = np.random.normal(m, sigma, N)[:, None]
x2 = alpha * x1 + nu
X = np.hstack((x1, x2))

newline = '\n'
print(f'Mean of input vector: {X.mean(axis=0)}')
# example output [4.99443765 9.99030127]
print(f'Theoretical prediction of the mean: {get_expected_mean()}')
# example output [ 5 10]
print(f'Covariance matrix of the input vector is {newline}{np.cov(X.T)}')
# example output 
# [[ 8.9983838  17.99611626]
# [17.99611626 38.24622575]]
print(f'The theoretical prediction of the covariance matrix is {newline}{get_expected_covariance()}')
# example output
# [[ 9.   18.  ]
#  [18.   38.25]]