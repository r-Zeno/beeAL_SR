import numpy as np

spikes_t_pn = np.load("/Users/zenorossi/Library/CloudStorage/OneDrive-Personal/noiseBeeSNN/sim_20250526_140348/pn_spike_t.npy")
spikes_id_pn = np.load("/Users/zenorossi/Library/CloudStorage/OneDrive-Personal/noiseBeeSNN/sim_20250526_140348/pn_spike_id.npy")

spikes_pn = np.stack((spikes_id_pn, spikes_t_pn),1)
spikes_pn2 = spikes_pn

n_number = np.unique(spikes_id_pn)

spikes_txid = dict()

for id, t in spikes_pn:
    key = int(id) # better to have it as int?
    
    if id not in spikes_txid:
        spikes_txid[key] = []
    
    spikes_txid[id].append(t)

for i in range(800):
    n = int(i)
    n_exist = spikes_txid.get(n)
    
    if n_exist is None:
        spikes_txid[n] = []
    
spikes_txid_sorted = dict(sorted(spikes_txid.items()))
    
spikes_txid2 = dict()

for id, t in spikes_pn2:
        key = int(id) # better to have it as int?
        
        if id not in spikes_txid2:
            spikes_txid2[key] = []
        
        spikes_txid2[id].append(t)

for i in range(800):
        n = int(i)
        n_exist = spikes_txid2.get(n)
        if n_exist is None:
            spikes_txid2[n] = []
        
spikes_txid_sorted2 = dict(sorted(spikes_txid2.items()))


spikes_2ns = dict()

if len(spikes_txid_sorted) == len(spikes_txid_sorted2):

    for i in range(len(spikes_txid_sorted)):

        key = int(i)
        spikes_2ns[key] = spikes_txid_sorted[key], spikes_txid_sorted2[key]