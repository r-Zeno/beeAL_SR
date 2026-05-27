import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, non_negative_factorization
from scipy.spatial.distance import euclidean, cosine

sim_dir_name = "sim_20260527_230853"
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sim_path = os.path.join(current_dir, "simulations", sim_dir_name)

res_od1_clean = np.load(os.path.join(sim_path, "odor_1", "sdf_clean.npy"))
res_od2_clean = np.load(os.path.join(sim_path, "odor_2", "sdf_clean.npy"))
res_od1_noisy = np.load(os.path.join(sim_path, "odor_1", "sdf_noisy.npy"))
res_od2_noisy = np.load(os.path.join(sim_path, "odor_2", "sdf_noisy.npy"))

dt = 1
baseline = 1000
settle_time = 250
start_idx = int(settle_time/dt)
end_idx = int(baseline/dt)

res_od1_clean = np.sqrt(res_od1_clean)
res_od2_clean = np.sqrt(res_od2_clean)
res_od1_noisy = np.sqrt(res_od1_noisy)
res_od2_noisy = np.sqrt(res_od2_noisy)

base_od1clean = np.mean(res_od1_clean[start_idx:end_idx,:], axis=0)
base_od2clean = np.mean(res_od2_clean[start_idx:end_idx,:], axis=0)
base_od1noisy = np.mean(res_od1_noisy[start_idx:end_idx,:], axis=0)
base_od2noisy = np.mean(res_od2_noisy[start_idx:end_idx,:], axis=0)

res_od1_clean = res_od1_clean - base_od1clean
res_od2_clean = res_od2_clean - base_od2clean
res_od1_noisy = res_od1_noisy - base_od1noisy
res_od2_noisy = res_od2_noisy - base_od2noisy
res_od1_clean = res_od1_clean[start_idx:-1,:]
res_od2_clean = res_od2_clean[start_idx:-1,:]
res_od1_noisy = res_od1_noisy[start_idx:-1,:]
res_od2_noisy = res_od2_noisy[start_idx:-1,:]

packaged = np.vstack([res_od1_clean, res_od2_clean, res_od1_noisy, res_od2_noisy])

pca = PCA(n_components=3)
pca.fit(packaged)
explained_var = np.sum(pca.explained_variance_ratio_) * 100
print(f"{explained_var:.2f}% of total variance")

traj_od1clean = pca.transform(res_od1_clean)
traj_od2clean = pca.transform(res_od2_clean)
traj_od1noisy = pca.transform(res_od1_noisy)
traj_od2noisy = pca.transform(res_od2_noisy)

dist_clean = [euclidean(traj_od1clean[t], traj_od2clean[t]) for t in range(len(traj_od1clean))]
dist_noisy = [euclidean(traj_od1noisy[t], traj_od2noisy[t]) for t in range(len(traj_od1noisy))]
distcos_clean = [cosine(traj_od1clean[t], traj_od2clean[t]) for t in range(len(traj_od1clean))]
distcos_noisy = [cosine(traj_od1noisy[t], traj_od2noisy[t]) for t in range(len(traj_od1noisy))]

tot_dist_clean = np.sum(dist_clean)
tot_dist_noisy = np.sum(dist_noisy)
tot_distcos_clean = np.sum(distcos_clean)
tot_distcos_noisy = np.sum(distcos_noisy)

plt.plot(dist_clean)
plt.plot(dist_noisy)
plt.show()

print(f"euclidean distance. clean: {tot_dist_clean}, noisy: {tot_dist_noisy}")
print(f"angular distance. clean: {tot_distcos_clean}, noisy: {tot_distcos_noisy}")


############### plotting
fig = plt.figure(figsize=(18, 6))
dt = 1.0
stim_idx = int(1000.0 / dt)

# helper to plot a single trajectory with markers
def plot_trajectory(ax, traj, color, linestyle, label):

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
            color=color, linestyle=linestyle, label=label, linewidth=2, alpha=0.8)
    
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
               color=color, marker='o', s=20)
    
    ax.scatter(traj[stim_idx, 0], traj[stim_idx, 1], traj[stim_idx, 2], 
               color=color, marker='*', s=150, edgecolors='black')

ax1 = fig.add_subplot(131, projection='3d')
plot_trajectory(ax1, traj_od1clean, color='blue', linestyle='-', label='odor 1 (noiseless)')
plot_trajectory(ax1, traj_od2clean, color='red', linestyle='-', label='odor 2 (noiseless)')
plot_trajectory(ax1, traj_od1noisy, color='cornflowerblue', linestyle='--', label='odor 1 (noisy)')
plot_trajectory(ax1, traj_od2noisy, color='salmon', linestyle='--', label='odor 2 (noisy)')

ax1.set_title('noiseless and noisy AL', fontsize=14)
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_zlabel('PC 3')
ax1.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

ax2 = fig.add_subplot(132, projection='3d')
plot_trajectory(ax2, traj_od1clean, color='blue', linestyle='-', label='odor 1 (noiseless)')
plot_trajectory(ax2, traj_od2clean, color='red', linestyle='-', label='odor 2 (noiseless)')

ax2.set_title('noiseless AL', fontsize=14)
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_zlabel('PC 3')
ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())
ax2.set_zlim(ax1.get_zlim())

ax3 = fig.add_subplot(133, projection='3d')
plot_trajectory(ax3, traj_od1noisy, color='cornflowerblue', linestyle='-', label='odor 1 (noisy)')
plot_trajectory(ax3, traj_od2noisy, color='salmon', linestyle='-', label='odor 2 (noisy)')

ax3.set_title('noisy AL', fontsize=14)
ax3.set_xlabel('PC 1')
ax3.set_ylabel('PC 2')
ax3.set_zlabel('PC 3')
ax3.set_xlim(ax1.get_xlim())
ax3.set_ylim(ax1.get_ylim())
ax3.set_zlim(ax1.get_zlim())

plt.tight_layout()
plt.show()
