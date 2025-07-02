import numpy as np
from matplotlib import pyplot as plt

t = np.linspace(1,3000,3000)

spikes_t_orn = np.load("/Users/zenorossi/Library/CloudStorage/OneDrive-Personal/noiseBeeSNN/sim_20250526_140348/orn_spike_t.npy")
spikes_id_orn = np.load("/Users/zenorossi/Library/CloudStorage/OneDrive-Personal/noiseBeeSNN/sim_20250526_140348/orn_spike_id.npy")
v_orn = np.load("/Users/zenorossi/Library/CloudStorage/OneDrive-Personal/noiseBeeSNN/sim_20250526_140348/orn_V_states.npy") 
or_rb_0 = np.load("/Users/zenorossi/Library/CloudStorage/OneDrive-Personal/noiseBeeSNN/sim_20250526_140348/or_rb_0_states.npy")
spikes_t_ln = np.load("/Users/zenorossi/Library/CloudStorage/OneDrive-Personal/noiseBeeSNN/sim_20250526_140348/ln_spike_t.npy")
spikes_id_ln = np.load("/Users/zenorossi/Library/CloudStorage/OneDrive-Personal/noiseBeeSNN/sim_20250526_140348/ln_spike_id.npy")
v_ln = np.load("/Users/zenorossi/Library/CloudStorage/OneDrive-Personal/noiseBeeSNN/sim_20250526_140348/ln_V_states.npy")

print(max(spikes_id_orn))
print(max(spikes_t_orn))
print(max(spikes_id_ln))
print(max(spikes_t_ln))

spikes_orn = np.stack((spikes_id_orn, spikes_t_orn),1)
spikes_ln = np.stack((spikes_id_ln, spikes_t_ln), 1)

num_or = max(spikes_id_orn)
num_ln = max(spikes_id_ln)

totake = spikes_orn[:, 0] == 5
spikes_orn_n200 = spikes_orn[totake, 1]

totake_ln = spikes_ln[:, 0] == 5
spikes_ln_n = spikes_ln[totake_ln, 1]

print(spikes_orn_n200)

fig, ax = plt.subplots()
ax.vlines(spikes_orn_n200, 0, 5)
plt.xlim([0,8000])
plt.show()

fig, ax = plt.subplots()
ax.vlines(spikes_ln_n, 0, 5)
plt.xlim([0,8000])
plt.show()

plt.plot(v_orn[:,5])
plt.show()

plt.plot(or_rb_0)
plt.show()

mean_v_orn = np.mean(v_orn, axis=1)
plt.plot(mean_v_orn)
plt.show()

mean_v_ln = np.mean(v_ln, axis=1)
plt.plot(mean_v_ln)
plt.show()

mean_per_n = np.mean(v_orn, axis=0)
mean_per_n_sorted = np.sort(-mean_per_n)

top5idx = np.argsort(-mean_per_n)[-5:] # get idx for top 5 "most depolarized" neurons across t
top5_n = spikes_orn[top5idx,:]

plt.plot(v_ln[4000:5000,5])
plt.show()

mean_ln = np.mean(v_ln[:,0:25], axis = 1)
plt.plot(mean_ln[0:1000])
plt.show()

plt.plot(mean_ln)
plt.show()

# top5idx = 
# for i in
#     plt.plot(v_orn[:,i])


    
