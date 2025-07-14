import numpy as np
import os

path = "/Users/zenorossi/beeAL/simulations/sim_20250712_202043/run_1"
paras = {

    "which_pop": ["pn"],

    "pop_number": 800,

    "which_metric": ["vp"],

    "odors": ["odor_1", "odor_2"]

}

# here dinamically retrieve data paths based on the number of folders, to allow to add more odors later on
# is it unnecessary?
odors_paths = {}
for odor in paras["odors"]:
    odors_paths[odor] = os.path.join(path, odor)

for odorpath in odors_paths:
    if os.path.isdir(odors_paths[odorpath]):
        print("directory as expected...")
    else: raise TypeError("Data folder structure is not what was expected!")
print("all good, starting distance analysis...")

spike_ts = []
spike_ids = []
for odor in odors_paths:

    for pop in paras["which_pop"]:

        curr_spike_t = np.load(os.path.join(odors_paths[odor], f"{pop}_spike_t.npy"))
        curr_spike_id = np.load(os.path.join(odors_paths[odor], f"{pop}_spike_id.npy"))

        spike_ts.append(curr_spike_t)
        spike_ids.append(curr_spike_id)
        
print(spike_ts)
print(spike_ids)

spikes_pop = []
for od_trial in range(len(spike_ts)):
    
    curr_idxt = np.stack((spike_ids[od_trial], spike_ts[od_trial]), 1)
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
    
    for i in range(paras["pop_number"]):
        
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




