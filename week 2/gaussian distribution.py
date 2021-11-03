import numpy as np
import matplotlib.pyplot as plt

N = int(10e6)
xx = np.random.randn(N)
plt.clf()

def gaussian_as_histogram(N: int, xx: np.array, bins=100) -> plt.hist:
    hist_stuff = plt.hist(xx, bins=bins)
    return hist_stuff

def plot_theoretical_gaussian(hist_stuff: plt.hist, N: int):
    # sum corners of histogram bars and divide by 2 to obtain centres
    bin_centres = 0.5 * (hist_stuff[1][1:] + hist_stuff[1][:-1])
    # obtain probability of datapoints to be at the bin centres
    pdf = np.exp(-0.5 * bin_centres**2) / np.sqrt(2 * np.pi)
    bin_width = bin_centres[1] - bin_centres[0]
    # from N given datapoints, p(x)*N*bin_width gives the number
    # of datapoints which will be found in a bin of width bin_width
    predicted_bin_heights = pdf * bin_width * N
    plt.plot(bin_centres, predicted_bin_heights)
    plt.show()

# plot_gaussian_as_histogram(N, xx)
hist_stuff = gaussian_as_histogram(N, xx)
plot_theoretical_gaussian(hist_stuff, N)
