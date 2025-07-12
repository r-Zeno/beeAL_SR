import numpy as np
import os

def _load_spikes(data_path):

    spikes_t = np.load(os.path.join(data_path, "pn_spike_t.npy"))
    spikes_id = np.load(os.path.join(data_path, "pn_spike_id.npy")) # very sad way to load, but ok for now
    spikes = np.stack((spikes_id, spikes_t), 1)

    return spikes

def _split_by_neuron(spikes, mod_paras):

    spikes_txid = dict()
    num_n = mod_paras["num"]["glo"]*mod_paras["num"]["pn"] # ugly but ok for now, later should pass just the number from the analyzer

    for id, t in spikes:
        key = int(id) # better to have it as int?
        
        if id not in spikes_txid:
            spikes_txid[key] = []
        
        spikes_txid[id].append(t)

    for i in range(num_n):
        n = int(i)
        n_exist = spikes_txid.get(n)
        
        if n_exist is None:
            spikes_txid[n] = []
    
    return spikes_txid

def couple_trains_by_neuron(data_path1, data_path2, mod_paras):
    """
    Takes 
    """

    spikes_1 = _load_spikes(data_path1)
    trains_1 = _split_by_neuron(spikes_1, mod_paras)

    spikes_2 = _load_spikes(data_path2)
    trains_2 = _split_by_neuron(spikes_2, mod_paras)

    trains_1_ord = dict(sorted(trains_1.items()))
    trains_2_ord = dict(sorted(trains_2.items()))
    coupled_spks = dict()

    if len(trains_1_ord) == len(trains_2_ord):

        for i in range(len((trains_1_ord))):

            key = int(i)
            coupled_spks[key] = (trains_1_ord[key], trains_2_ord[key])
    else: raise ValueError("Neuron number for the spk trains doesn't match!")

    return coupled_spks
