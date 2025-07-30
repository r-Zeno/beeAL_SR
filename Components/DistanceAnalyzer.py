import numpy as np
import os
import time
from joblib import Parallel, delayed
from helpers import data_prep4numba_distance

class DistanceAnalyzer:

    def __init__(self, data_path, an_paras:dict, neurons_idx):

        self.path = data_path
        self.paras = an_paras
        self.neurons_idx = neurons_idx
        self.cost_vp = self.paras["cost_vp"]
        # here dinamically retrieve data paths based on the number of folders, to allow to add more odors later on
        # is it unnecessary?
        odors_paths = {}
        for odor in self.paras["odors"]:
            odors_paths[odor] = os.path.join(self.path, odor)

        for odorpath in odors_paths:
            if os.path.isdir(odors_paths[odorpath]):
                print("directory as expected...")
            else: raise TypeError("Data folder structure is not what was expected!")
        print("all good, starting distance analysis...")

        self.spike_ts = {}
        self.spike_ids = {}
        i = 0
        for odor in odors_paths:
            i += 1
            curr_od = f"od_{i}"
            self.spike_ts[curr_od] = {}
            self.spike_ids[curr_od] = {}

            for pop in self.paras["which_pop"]:

                curr_spike_t = np.load(os.path.join(odors_paths[odor], f"{pop}_spike_t.npy"))
                curr_spike_id = np.load(os.path.join(odors_paths[odor], f"{pop}_spike_id.npy"))

                self.spike_ts[curr_od][pop] = curr_spike_t
                self.spike_ids[curr_od][pop] = curr_spike_id
                
        print(self.spike_ts)
        print(self.spike_ids)

    def _coupler(self):

        spikes_pop = {}
        for od, pops in self.spike_ts.items():
            spikes_pop[od] = {}

            for pop, spike_ts in pops.items():

                spike_ids = self.spike_ids[od][pop]

                time_select = (spike_ts >= 1000) & (spike_ts <= 4000) # only take spikes during odor stim
                filtered_ts = spike_ts[time_select]
                filtered_ids = spike_ids[time_select]
                curr_idxt = np.stack((filtered_ids, filtered_ts), 1)

                spikes_pop[od][pop] = curr_idxt

        spikes_idxt = {}
        i = 0
        for od, pops in spikes_pop.items():
            spikes_idxt[od] = {}
            
            for pop, joined_spks in pops.items():
                spikes_idxt[od][pop] = {}

                for id, t in joined_spks:
                    key = int(id)
                    
                    if key not in spikes_idxt[od][pop]:
                        spikes_idxt[od][pop][key] = []
                        
                    spikes_idxt[od][pop][key].append(t)

        for od, pops in spikes_idxt.items():
            
            for pop, ids in pops.items():

                for i in range(self.paras["which_pop"][pop][1]):
                    
                    n_id = int(i)
                    n_id_exist = spikes_idxt[od][pop].get(n_id)
                    
                    if n_id_exist is None:
                        spikes_idxt[od][pop][n_id] = []
                
                spikes_idxt[od][pop] = dict(sorted(spikes_idxt[od][pop].items()))

        spikes_coupled = {}
        for pop in spikes_idxt["od_1"]:
            spikes_coupled[pop] = {}

            for i in range(len(spikes_idxt["od_1"][pop])):
                key = int(i)
                spikes_coupled[pop][key] = (spikes_idxt["od_1"][pop][key], spikes_idxt["od_2"][pop][key])

        spikes_coupled_corr = {} # start by checking this
        for pop in spikes_coupled:
            spikes_coupled_corr[pop] = {}

            for pop_idx, idxs in self.neurons_idx.items():

                for idx in idxs:
                    spikes_coupled_corr[pop][idx] = spikes_coupled[pop][idx]

        print(f"analyzing distance for {len(spikes_coupled_corr)} neurons")
        return spikes_coupled_corr

    def _analyzer(self, spikes_coupled):

        # running the vp algorithm in parallel across neurons on cpu cores
        print("Starting parallel computation of VP distance...")
        start = time.time()
        vp_dist = Parallel(n_jobs=-1)(delayed(data_prep4numba_distance)(spikes_coupled[n][0],spikes_coupled[n][1],self.cost_vp) for n in spikes_coupled)
        end = time.time()
        timetaken = round(end-start, 2)
        print(f"All distance values computed, it took {timetaken}s")
        mean_vpdist = np.mean(vp_dist)

        return mean_vpdist, vp_dist # for debugging
    
    def compute_distance(self):

        spikes_coupled = self._coupler()
        mean_vpdist, vp_dist = self._analyzer(spikes_coupled)
        return mean_vpdist, vp_dist
