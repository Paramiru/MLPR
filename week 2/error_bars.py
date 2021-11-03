import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sem

# array of 100 random numbers
xx = 1 * (np.random.rand(100) < 0.3)

def estimate_mean_error(xx: np.array) -> float:
    N = xx.size
    # numpy's method is already the unbiased estimate (divides by N-1)
    unbiased_standard_deviation_estimate = xx.std()
    return unbiased_standard_deviation_estimate / np.sqrt(N)

print(sem(xx))
print(estimate_mean_error(xx))


