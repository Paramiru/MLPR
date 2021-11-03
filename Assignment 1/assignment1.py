import numpy as np
import matplotlib.pyplot as plt

amp_data = np.load('amp_data.npz')['amp_data'][:, None]
times = np.arange(len(amp_data))

plt.clf()
plt.plot(times, amp_data)
# plt.axis([-0.4, 0.6, 0, 80000])
# plt.hist(amp_data, bins=30000)
plt.show()
