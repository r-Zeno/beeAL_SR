import numpy as np

class RateAnalyzer:

    def __init__(self, rates:dict, neuronsidx, paras:dict):

        self.rates = rates
        self.paras = paras
        self.idxn = neuronsidx

    def _rate_measures_compute(self): # duplicate of _select_neurons logic, but would be messy to save delta rates from NeuronSelector

        runs = list(self.rates["baseline"].keys())
        n_runs = len(runs)
        n_neurons = self.paras["pop_number"]
        odors = self.paras["odors"]
        pops = self.paras["which_pop"]

        rate_delta = np.zeros((n_neurons, n_runs), dtype=float)
        relative_rate_delta = np.zeros((n_neurons, n_runs), dtype=float)
        
        run_idx = {}
        for i, run in enumerate(runs):
            run_idx[run] = i

        for run in runs:
            for odor in odors:
                for pop in pops:
                    for neuron in range(n_neurons):

                        curr_base_rate = self.rates["baseline"][run][odor][pop][neuron]
                        curr_stim_rate = self.rates["stimulation"][run][odor][pop][neuron]
                        curr_delta = abs(curr_base_rate - curr_stim_rate)
                        if curr_base_rate == 0 and curr_stim_rate == 0:
                            curr_rel_delta = 0.0
                        else: curr_rel_delta = (curr_stim_rate - curr_base_rate)/(curr_stim_rate + curr_base_rate)

                        idx = run_idx[run]
                        rate_delta[neuron, idx] = curr_delta
                        relative_rate_delta[neuron, idx] = curr_rel_delta

        return rate_delta, relative_rate_delta

    def get_rate_diff(self):

        rate_delta, relative_rate_delta = self._rate_measures_compute()
        return rate_delta, relative_rate_delta
