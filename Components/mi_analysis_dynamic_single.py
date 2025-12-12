import numpy as np
import os
from sklearn.feature_selection import mutual_info_regression

class MiDynamicSingle:

    def __init__(self, stim_path, paras):
        ZeroDivisionError()
        
        self.k = int(paras["k_neighbors"])
        self.stim_path = stim_path

    def _neuron_extract_spikes(self):
        ZeroDivisionError()

    def _neuron_smooth_rate():
        ZeroDivisionError()
    
    def Analysis(self):

        n_spks = self._neuron_extract_spikes()
        n_rate = self._neuron_smooth_rate(n_spks)
        stim = self.stim_path

        mi = mutual_info_regression(n_rate, stim, n_neighbors = self.k, n_jobs = -1)

        return mi
