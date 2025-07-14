import numpy as np
from matplotlib import pyplot as plt

dist_means = np.load("/Users/zenorossi/beeAL/simulations/sim_20250714_201142/mean_vp_dist_x_noiselvls.npy")
dist_single = np.load("/Users/zenorossi/beeAL/simulations/sim_20250714_201142/single_vp_dist_values.npy")
print(dist_means)

plt.hist(dist_single[0])
plt.show()

plt.hist(dist_single[1])
plt.show()

plt.hist(dist_single[2])
plt.show()

x = [0, 4.5, 9]
y = [0.27440597, 0.93790259, 0.79402039]

plt.scatter(x, y)
plt.show()