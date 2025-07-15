import numpy as np
from matplotlib import pyplot as plt

dist_means = np.load("/Users/zenorossi/beeAL/simulations/sim_20250715_115946/mean_vp_dist_x_noiselvls.npy")
print(dist_means)

# x = dist_means
# y = np.linspace(0,10,1000)

# plt.scatter(y,x)
# plt.savefig("/home/zeno/beeAL_SR/simulations/sim_20250715_000548/output.jpg")