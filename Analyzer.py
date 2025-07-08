import numpy as np
from matplotlib import pyplot as plt
import os
from helpers import make_sdf, force_aspect, glo_avg

class Analyzer:
    
    def __init__(self, path, an_paras:dict, model_paras:dict):
        
        self.path = path
        self.an_paras = an_paras
        self.pops = self.an_paras["pop_to_analyze"]
        self.mod_paras = model_paras
        self.toplot = self.an_paras["plot"]
        self.save1 = self.an_paras["save_sdf"]
        self.save2D = self.an_paras["save_2Dplot"]
        self.save3D = self.an_paras["save_3Dplot"]

    def _data_collector(self):
        
        pops = self.an_paras["pop_to_analyze"]
        nums = self.mod_paras["num"]

        anpop = []
                
        for pop, toanalyze in pops.items():
            
            if toanalyze:
                anpop.append(pop)

        # hardcoded values for the spike density function        
        sigma_sdf = 100
        dt_sdf = 1

        # harcoded values from experiment
        total_t = 8000

        for pop in anpop:
            
            N = nums["glo"]*nums[pop]
            spike_t = np.load(os.path.join(self.path, f"{pop}_spike_t.npy"))
            spike_id = np.load(os.path.join(self.path, f"{pop}_spike_id.npy"))
            
            lsdfs_od1 = []
            gsdfs_od1 = []
            
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
            
            if self.save1:
                np.save(f"{pop}_glo_avg_sdf", gsdfs_od1)
            
            if self.toplot:
                mn= [-5, -40]
                mx= [40, 200]
                ts1 = np.transpose(gsdfs_od1[0]) # array must be inverse (neurons x time)
                
                fig, ax = plt.subplots()
                ax.imshow(ts1, vmin=mn[0], vmax=mx[0], cmap="hot")
                force_aspect(ax,0.8)
                
                if self.save2D:
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
                
                if self.save3D:
                    plt.savefig(f"{pop}_sdf_surf.png", dpi=300)
