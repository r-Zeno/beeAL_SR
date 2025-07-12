import numpy as np
from matplotlib import pyplot as plt
import os
from helpers import make_sdf, force_aspect, glo_avg

class SDFplotter:
    """
    Responsible for creating an instance of sdf analysis/plotting, need to explicitly call '.analyze()'
    """
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
        self.nums = self.mod_paras["num"]

        self.anpop = []
                
        for pop, toanalyze in pops.items():
            
            if toanalyze:
                self.anpop.append(pop)

    def _sdf_maker(self):

        sdf_path = os.path.join(self.path, "sdf")
        os.makedirs(sdf_path, exist_ok=True)

        # hardcoded values for the spike density function        
        sigma_sdf = 100
        dt_sdf = 1
        self.pops_gsdfs_od1 = {}

        # harcoded values from experiment
        total_t = 8000

        od_1_path = os.path.join(self.path, "odor_1")

        if os.path.isdir(od_1_path):

            total_t = 5000 # not ideal overwriting of sim time to adapt to the separate exp duration, should get it from the parameters
            # or passed from the simulator
            self.pops_gsdfs_od2 = {}
            print("Folder structure suggests that this is the separeted odors exp, if you don't know what this means check!")

            for odor_lbl in ["odor_1", "odor_2"]:

                target_dict = self.pops_gsdfs_od1 if odor_lbl == "odor_1" else self.pops_gsdfs_od2
                work_path = os.path.join(self.path, odor_lbl)

                for pop in self.anpop:

                    n = self.nums["glo"]*self.nums[pop]
                    spike_t = np.load(os.path.join(work_path, f"{pop}_spike_t.npy"))
                    spike_id = np.load(os.path.join(work_path, f"{pop}_spike_id.npy"))

                    lsdfs_od1 = []
                    gsdfs_od1 = []
                    li = 0
                    left = 0
                    right = total_t
                    
                    while li < len(spike_t) and spike_t[li] < left:
                        li += 1
                    ri = li
                    while ri < len(spike_t) and spike_t[ri] < right:
                        ri += 1
                    lsdfs_od1.append(make_sdf(spike_t[li:ri], spike_id[li:ri], np.arange(0,n), left-3*sigma_sdf, right-3*sigma_sdf, dt_sdf, sigma_sdf))
                    gsdfs_od1.append(glo_avg(lsdfs_od1[-1],60))

                    target_dict[pop] = gsdfs_od1

                    if self.save1:
                        save_dir = os.path.join(sdf_path, odor_lbl)
                        os.makedirs(save_dir, exist_ok=True)
                        np.save(os.path.join(sdf_path, f"{pop}_glo_avg_sdf"), gsdfs_od1)
        else:
            print("Folder structure suggests that this is the consecutive odors exp, if you don't know what this means check!")
            for pop in self.anpop:
                
                n = self.nums["glo"]*self.nums[pop]
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
                while ri < len(spike_t) and spike_t[ri] < right: # incrementing ri till at the end of the trial time window
                    ri += 1
                lsdfs_od1.append(make_sdf(spike_t[li:ri], spike_id[li:ri], np.arange(0,n), left-3*sigma_sdf, right-3*sigma_sdf, dt_sdf, sigma_sdf)) # originally it was "left-3*sigma_sdf" and so on for the limits,
                # changed it because its already done within the make_sdf code itself. could this be a potential issue? need to check!
                gsdfs_od1.append(glo_avg(lsdfs_od1[-1],60))

                self.pops_gsdfs_od1[pop] = gsdfs_od1
                
                if self.save1:
                    np.save(os.path.join(sdf_path, f"{pop}_glo_avg_sdf"), gsdfs_od1)
            
    def _plotter(self):

        plot_path = os.path.join(self.path, "plots")
        os.makedirs(plot_path, exist_ok=True)

        for pop in self.anpop:

            data2plot = []

            if pop in self.pops_gsdfs_od1:
                data2plot.append(("od1", self.pops_gsdfs_od1[pop]))
            
            if hasattr(self, 'pops_gsdfs_od2') and pop in self.pops_gsdfs_od2:
                data2plot.append(("od2", self.pops_gsdfs_od2[pop]))
        
            for lbl, gsdfs in data2plot:

                mn= [-5, -40]
                mx= [40, 200]

                ts1 = np.transpose(gsdfs[0]) # array must be inverse (neurons x time)
                    
                fig, ax = plt.subplots()
                ax.imshow(ts1, vmin=mn[0], vmax=mx[0], cmap="hot")
                force_aspect(ax,0.8)
                        
                if self.save2D:
                    plt.savefig(os.path.join(plot_path, f"{pop}_{lbl}_sdf_hotmap.png"), dpi=300)
                plt.close(fig)
                
                x = np.arange(np.size(ts1,0))
                y = np.arange(np.size(ts1,1))
                X, Y = np.meshgrid(x, y, indexing= 'ij')
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111, projection = '3d')
                ax2.grid(False)
                ax2.xaxis.pane.fill = False
                ax2.yaxis.pane.fill = False
                ax2.zaxis.pane.fill = False
                ax2.xaxis.pane.set_edgecolor('white')
                ax2.yaxis.pane.set_edgecolor('white')
                ax2.zaxis.pane.set_edgecolor('white')
                ax2.plot_surface(X,Y,ts1, cmap = 'viridis')

                if self.save3D:
                    plt.savefig(os.path.join(plot_path, f"{pop}_{lbl}_sdf_surf.png"), dpi=300)
                plt.close(fig2)

    def plot(self):
        """
        Colled to compute and plot spike density functions for pops specified in the config json 
        """
        print("Starting analysis...")

        self._data_collector()
        self._sdf_maker()
        if self.toplot:
            self._plotter()

        print(f"Computed sdf for {self.anpop}. plot: {self.toplot}, sdf saved: {self.save1}")
