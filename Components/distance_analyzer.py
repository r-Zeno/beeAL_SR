import numpy as np
import os
import time
import numba
from numba import types
from numba.typed import dictobject
from joblib import Parallel, delayed
from helpers import vp_metric

class DistanceAnalyzer:

    def __init__(self, data_path, an_paras:dict):

        self.path = data_path
        self.paras = an_paras
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

            for pop in self.paras["which_pop"]:

                curr_spike_t = np.load(os.path.join(odors_paths[odor], f"{pop}_spike_t.npy"))
                curr_spike_id = np.load(os.path.join(odors_paths[odor], f"{pop}_spike_id.npy"))

                self.spike_ts.append(curr_spike_t)
                self.spike_ids.append(curr_spike_id)
                
        print(self.spike_ts)
        print(self.spike_ids)

    def _coupler(self):

        spikes_pop = []
        for od_trial in range(len(self.spike_ts)):
            
            curr_idxt = np.stack((self.spike_ids[od_trial], self.spike_ts[od_trial]), 1)
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
            
            for i in range(self.paras["pop_number"]):
                
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

        return spikes_coupled

    def _analyzer(self, spikes_coupled):

        # running the vp algorithm in parallel across neurons on cpu cores
        print("Starting parallel computation of VP distance...")

        start = time.time()
        vp_dist = Parallel(n_jobs=-1)(delayed(vp_metric)(spikes_coupled[n][0],spikes_coupled[n][1],self.cost_vp) for n in spikes_coupled)
        end = time.time()
        timetaken = round(end-start, 2)
        print(f"All distance values computed, it took {timetaken}s")
        mean_vpdist = np.mean(vp_dist)

        return mean_vpdist, vp_dist # for debugging
    
    #def _dict_to_typed_dict_converter(self, dict2convert):


    def compute_distance(self):

        spikes_coupled = self._coupler()
        mean_vpdist, vp_dist = self._analyzer(spikes_coupled)
        return mean_vpdist, vp_dist
