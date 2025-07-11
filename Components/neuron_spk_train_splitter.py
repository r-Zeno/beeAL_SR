import numpy as np
import os

class NeuronSplitter:

    def __init__(self, data_path):

        self.spikes_t = np.load(os.path.join(data_path, "pn_spike_t.npy"))
        self.spikes_id = np.load(os.path.join(data_path, "pn_spike_id.npy")) # very sad way to load, but ok for now

        self.spikes = np.stack((self.spikes_id, self.spikes_t), 1)

    def _splitter(self):

        

