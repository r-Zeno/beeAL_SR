import numpy as np
from matplotlib import pyplot as plt

dist_means = np.load("/home/zeno/beeAL_SR/simulations/sim_20250715_222939/mean_vp_dist_x_noiselvls.npy")
print(dist_means)

x = dist_means
y = np.linspace(0,6,1000)

plt.scatter(y,x)
plt.savefig("/home/zeno/beeAL_SR/simulations/sim_20250715_222939/outputlow.jpg")