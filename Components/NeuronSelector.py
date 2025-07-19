import os
import sys
import numpy as np

class NeuronSelector:

    def __init__(self, paras, rates, rate_delta_odors_diff, noise_lvls, how2select:bool):

        self.rates = rates
        self.paras = paras
        self.how2select = how2select
        self.rate_delta_odors_diff = rate_delta_odors_diff
        self.noise_lvls = noise_lvls

    def _select_neurons(self, rates:dict):

        runs = list(rates["baseline"].keys())
        n_runs = len(runs)

        n_neurons = self.paras["pop_number"]
        odors = self.paras["odors"]
        pops = self.paras["which_pop"]

        decision_matrix = np.zeros((n_neurons, n_runs), dtype=bool)
        
        run_idx = {}
        for i, run in enumerate(runs):
            run_idx[run] = i
        
        for run in runs:
            for odor in odors:
                for pop in pops:
                    for neuron in range(n_neurons):

                        curr_baseline_rate = rates["baseline"][run][odor][pop][neuron]
                        curr_stimulation_rate = rates["stimulation"][run][odor][pop][neuron]

                        if curr_stimulation_rate > (curr_baseline_rate + self.paras["threshold"]):

                            idx = run_idx[run]
                            decision_matrix[neuron, idx] = True

        responsive_ns = np.where(np.any(decision_matrix, axis=1))[0]
        neurons2analyze = responsive_ns.tolist()

        return neurons2analyze, decision_matrix

    def _select_neurons_0noise(self, rates:dict, isexclusive:bool):

        n_neurons = self.paras["pop_number"]
        odors = self.paras["odors"]
        pops = self.paras["which_pop"]
        run_idx = "run_0" # assuming that the first run is at 0 noise

        if isexclusive:
            n_odors = len(odors)
            odor_map = {}
            for i, odor in enumerate(odors):
                odor_map[odor] = i
            decision_vector = np.zeros((n_neurons, n_odors), dtype=bool)
        else: decision_vector = np.zeros(n_neurons, dtype=bool)
        
        for odor in odors:
            for pop in pops:
                for neuron in range(n_neurons):

                    curr_baseline_rate = rates["baseline"][run_idx][odor][pop][neuron]
                    curr_stimulation_rate = rates["stimulation"][run_idx][odor][pop][neuron]

                    if curr_stimulation_rate > (curr_baseline_rate + self.paras["threshold_0noise"]): # at 0 noise only change if touched by odor

                        if isexclusive:
                            odor_idx = odor_map[odor]
                            decision_vector[neuron, odor_idx] = True
                        else:
                            decision_vector[neuron] = True

        if isexclusive:
            responsive_idx = np.all(decision_vector, 1) # taking only neurons that where responsive (at 0 noise) to both odors
        else: responsive_idx = decision_vector

        responsive_ns = np.where(responsive_idx)[0]
        neurons2analyze = responsive_ns.tolist()

        return neurons2analyze, decision_vector
    
    def _take_odor_selective(self, rate_delta_odors_diff, noise_lvls):

        th = self.paras["odor_diff_threshold(Hz)"]
        nlvl = self.paras["reference_noiselvl"]
        ndiff = np.abs(noise_lvls - nlvl)
        idxn = ndiff.argmin()
        print(f"sanity check: taking run {idxn} as reference")
        run2take = rate_delta_odors_diff[:, idxn]
        neurons2analyze = np.where(run2take >= th)[0]
        neurons2analyze.tolist()

        return neurons2analyze
        
    def select(self):

        if self.how2select["only0noise"]:
            neurons2analyze, _ = self._select_neurons_0noise(self.rates, self.paras["isexclusive"])
        elif self.how2select["select_overall"]:
            neurons2analyze, _ = self._select_neurons(self.rates)
        elif self.how2select["take_odor_selectives"]:
            neurons2analyze = self._take_odor_selective(self.rate_delta_odors_diff, self.noise_lvls)
        else: raise ValueError("Must choose a selection criterion for the nuerons to use in dist analysis!")

        return neurons2analyze
    