import numpy as np
import os
import time
from joblib import Parallel, delayed
from helpers import data_prep4numba_distance

class DistanceAnalyzer:

    def __init__(self, data_path, an_paras:dict, pop, neurons_idx):

        self.path = data_path
        self.paras = an_paras
        self.neurons_idx = neurons_idx
        self.pop = pop
        self.n_neurons = self.paras["which_pop"][self.pop][1]
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

        self.spike_ts = []
        self.spike_ids = []
        for odor in odors_paths:

            curr_spike_t = np.load(os.path.join(odors_paths[odor], f"{self.pop}_spike_t.npy"))
            curr_spike_id = np.load(os.path.join(odors_paths[odor], f"{self.pop}_spike_id.npy"))

            self.spike_ts.append(curr_spike_t)
            self.spike_ids.append(curr_spike_id)
                
        print(self.spike_ts)
        print(self.spike_ids)

    def _coupler(self):

        spikes_pop = []
        for od_trial in range(len(self.spike_ts)):
            
            spike_ts = self.spike_ts[od_trial]
            spike_ids = self.spike_ids[od_trial]

            time_mask = (spike_ts >= 1000) & (spike_ts <= 4000)
            filtered_ts = spike_ts[time_mask]
            filtered_ids = spike_ids[time_mask]
            curr_idxt = np.stack((filtered_ids, filtered_ts), 1)
            spikes_pop.append(curr_idxt)

        spikes_idxt = {}
        i = 0
        for od_array in spikes_pop:
            i += 1
            spikes_idxt[f"od_{i}"] = {}
            
            for id, t in od_array:
                key = int(id)
                
                if id not in spikes_idxt[f"od_{i}"]:
                    spikes_idxt[f"od_{i}"][key] = []
                    
                spikes_idxt[f"od_{i}"][key].append(t)

        for od_dict in spikes_idxt:
            
            for i in range(self.n_neurons):
                
                n_id = int(i)
                n_id_exist = spikes_idxt[od_dict].get(n_id)
                
                if n_id_exist is None:
                    spikes_idxt[od_dict][n_id] = []
            
            spikes_idxt[od_dict] = dict(sorted(spikes_idxt[od_dict].items()))

        spikes_coupled = {}
        if len(spikes_idxt["od_1"]) == len(spikes_idxt["od_2"]):
                
            for i in range(len(spikes_idxt["od_1"])):
                key = int(i)
                spikes_coupled[key] = (spikes_idxt["od_1"][key], spikes_idxt["od_2"][key])
        else: raise TypeError("Neuron number for the 2 runs does not match, something went very wrong!")

        spikes_coupled_corr = {}
        for idx in self.neurons_idx:
            spikes_coupled_corr[idx] = spikes_coupled[idx]

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
