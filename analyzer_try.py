import numpy as np
from matplotlib import pyplot as plt
from helpers import make_sdf, force_aspect, glo_avg
import os
import json


plot = True
save_plot2D = True
save_plot3D = True

path = "/Users/zenorossi/beeAL/sim_20250707_203605"
with open("/Users/zenorossi/beeAL/analysis_params.json") as f:
    paras = json.load(f)
    
with open("/Users/zenorossi/beeAL/example_for_modelbuilds.json") as f:
    mod_paras = json.load(f)

pops = paras["pop_to_analyze"]
nums = mod_paras["num"]

anpop = []
        
for pop, toanalyze in pops.items():
    
    if toanalyze:
        anpop.append(pop)

# hardcoded values for the spike density function        
sigma_sdf = 100
dt_sdf = 1

# harcoded values from experiment
total_t = 8000
trials = [1,2]

for pop in anpop:
    
    N = nums["glo"]*nums[pop]
    spike_t = np.load(os.path.join(path, f"{pop}_spike_t.npy"))
    spike_id = np.load(os.path.join(path, f"{pop}_spike_id.npy"))
    
    lsdfs_od1 = []
    gsdfs_od1 = []
    lsdfs_od2 = []
    gsdfs_od2 = []
    
    li = 0
    
    left = 0
    right = total_t
    
    while li < len(spike_t) and spike_t[li] < left: # incrementing li till at the start of the trial time window
            li += 1
    ri = li
    while ri <len(spike_t) and spike_t[ri] < right: # incrementing ri till at the end of the trial time window
            ri += 1
    lsdfs_od1.append(make_sdf(spike_t[li:ri], spike_id[li:ri], np.arange(0,N), left-3*sigma_sdf, right-3*sigma_sdf, dt_sdf, sigma_sdf)) # originally it was "left-3*sigma_sdf" and so on for the limits,
    # i changed it beacause its already done within the make_sdf code itself. could this be a potential issue? need to check!
    gsdfs_od1.append(glo_avg(lsdfs_od1[-1],60))
    
    if plot:
        mn= [-5, -40]
        mx= [40, 200]
        ts1 = np.transpose(gsdfs_od1[0]) # array must be inverse (neurons x time)
        
        fig, ax = plt.subplots()
        ax.imshow(ts1, vmin=mn[0], vmax=mx[0], cmap="hot")
        force_aspect(ax,0.8)
        
        if save_plot2D:
            plt.savefig(f"{pop}_sdf_hotmap.png", dpi=300)
        
        x = np.arange(np.size(ts1,0))
        y = np.arange(np.size(ts1,1))
        X, Y = np.meshgrid(x, y, indexing= 'ij')
        fig = plt.figure()
        ax2 = fig.add_subplot(111, projection = '3d')
        ax2.grid(False)
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.xaxis.pane.set_edgecolor('white')
        ax2.yaxis.pane.set_edgecolor('white')
        ax2.zaxis.pane.set_edgecolor('white')
        ax2.plot_surface(X,Y,ts1, cmap = 'viridis')
        
        if save_plot3D:
            plt.savefig(f"{pop}_sdf_surf.png", dpi=300)



