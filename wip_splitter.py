import numpy as np

spikes_t_pn = np.load("/Users/zenorossi/Library/CloudStorage/OneDrive-Personal/noiseBeeSNN/sim_20250526_140348/pn_spike_t.npy")
spikes_id_pn = np.load("/Users/zenorossi/Library/CloudStorage/OneDrive-Personal/noiseBeeSNN/sim_20250526_140348/pn_spike_id.npy")

spikes_pn = np.stack((spikes_id_pn, spikes_t_pn),1)

n_number = np.unique(spikes_id_pn)

spikes_txid = dict()

for id, t in spikes_pn:
    key = int(id)
    
    if id not in spikes_txid:
        spikes_txid[key] = []
    
    spikes_txid[id].append(t)
    

    