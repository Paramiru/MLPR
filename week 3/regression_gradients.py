import numpy as np

N = 100000
D = 30

X = np.random.randn(N,D)
yy = np.random.randn(N)

result1 = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, yy))
result2 = np.dot(np.linalg.solve(np.dot(X.T, X), X.T), yy)   # ?

print(result1)
print(result2)
print(np.isclose(result1, result2))