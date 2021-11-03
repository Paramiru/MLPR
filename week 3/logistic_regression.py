import numpy as np

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def sigmoid_derivative(a):
    return sigmoid(a) * (1 - sigmoid(a))

def approx_sigmoid_derivative(a, epsilon=10e-5):
    return (sigmoid(a + epsilon/2) - sigmoid(a - epsilon/2)) / epsilon

def approx_sigmoid_derivative_2(a, epsilon=10e-5):
    """Approximate derivative of analytical function using complex numbers"""
    return sigmoid(a + epsilon * 1j).imag / epsilon

def main():
    N = 5
    a = np.random.normal(0, 10, size=N)[:,None]
    print(sigmoid(a))
    print(sigmoid_derivative(a))
    print(approx_sigmoid_derivative(a))
    print(approx_sigmoid_derivative_2(a, 1e-9))

if __name__ == "__main__":
    main()
