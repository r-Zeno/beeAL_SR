import numpy as np

class RateAnalyzer:

    def __init__(self, rates:dict, paras:dict):

        self.rates = rates
        self.paras = paras

    def _rate_measures_compute(self): # duplicate of _select_neurons logic, but would be messy to save delta rates from NeuronSelector

        runs = list(self.rates["baseline"].keys())
        n_runs = len(runs)
        n_neurons = self.paras["pop_number"]
        odors = self.paras["odors"]
        pops = self.paras["which_pop"]

        rate_deltas = {}
        relative_rate_deltas = {}
        flat_rate_base = {}
        flat_rate_stim = {}
        for odor in odors:
            rate_deltas[odor] = np.zeros((n_neurons, n_runs), dtype=float)
            relative_rate_deltas[odor] = np.zeros((n_neurons, n_runs), dtype=float)
            flat_rate_base[odor] = np.zeros((n_neurons, n_runs), dtype=float)
            flat_rate_stim[odor] = np.zeros((n_neurons, n_runs), dtype=float)

        run_idx = {}
        for i, run in enumerate(runs):
            run_idx[run] = i

        # this could be vectorized, but won't do it now
        for run in runs:
            for odor in odors:
                for pop in pops: # ! delta values would get overwritten if more than 1 pop !
                    for neuron in range(n_neurons):

                        curr_base_rate = self.rates["baseline"][run][odor][pop][neuron]
                        curr_stim_rate = self.rates["stimulation"][run][odor][pop][neuron]

                        curr_delta = abs(curr_base_rate - curr_stim_rate)
                        if curr_base_rate == 0 and curr_stim_rate == 0:
                            curr_rel_delta = 0.0
                        else: curr_rel_delta = (curr_stim_rate - curr_base_rate)/(curr_stim_rate + curr_base_rate) # avoids dividing by 0

                        idx = run_idx[run]
                        rate_deltas[odor][neuron, idx] = curr_delta
                        relative_rate_deltas[odor][neuron, idx] = curr_rel_delta

                        flat_rate_base[odor][neuron, idx] = curr_base_rate
                        flat_rate_stim[odor][neuron, idx] = curr_stim_rate

        return rate_deltas, relative_rate_deltas, flat_rate_base, flat_rate_stim
    
    def _rate_measures_odordiff_compute(self):

        runs = list(self.rates["baseline"].keys())
        n_runs = len(runs)
        n_neurons = self.paras["pop_number"]
        odors = self.paras["odors"]
        pops = self.paras["which_pop"]

        rate_delta_odors = np.zeros((n_neurons, n_runs), dtype=float)
        relative_rate_delta_odors = np.zeros((n_neurons, n_runs), dtype=float)

        run_idx = {}
        for i, run in enumerate(runs):
            run_idx[run] = i

        # this could be vectorized, but won't do it now
        for run in runs:
            for pop in pops: # ! delta values would get overwritten if more than 1 pop !
                for neuron in range(n_neurons):

                    curr_base_rate_od1 = self.rates["baseline"][run]["odor_1"][pop][neuron]
                    curr_stim_rate_od1 = self.rates["stimulation"][run]["odor_1"][pop][neuron]
                    curr_base_rate_od2 = self.rates["baseline"][run]["odor_2"][pop][neuron]
                    curr_stim_rate_od2 = self.rates["stimulation"][run]["odor_2"][pop][neuron]

                    curr_delta_odors = abs(curr_stim_rate_od1 - curr_stim_rate_od2)

                    if curr_base_rate_od1 == 0 and curr_stim_rate_od1 == 0:
                        curr_rel_delta_od1 = 0.0
                    else: curr_rel_delta_od1 = (curr_stim_rate_od1 - curr_base_rate_od1)/(curr_stim_rate_od1 + curr_base_rate_od1)
                    if curr_base_rate_od2 == 0 and curr_stim_rate_od2 == 0:
                        curr_rel_delta_od2 = 0.0
                    else: curr_rel_delta_od2 = (curr_stim_rate_od2 - curr_base_rate_od2)/(curr_stim_rate_od2 + curr_base_rate_od2)
                    curr_rel_delta_odors = abs(curr_rel_delta_od1 - curr_rel_delta_od2) # don't care about the direction of the difference (od1/od2)

                    idx = run_idx[run]
                    rate_delta_odors[neuron, idx] = curr_delta_odors
                    relative_rate_delta_odors[neuron, idx] = curr_rel_delta_odors

        return rate_delta_odors, relative_rate_delta_odors

    def get_rate_diffs(self):

        rate_delta, relative_rate_delta, flat_rate_base, flat_rate_stim = self._rate_measures_compute()
        rate_delta_odorsdiff, relative_rate_delta_odorsdiff = self._rate_measures_odordiff_compute()

        return flat_rate_base, flat_rate_stim, rate_delta, relative_rate_delta, rate_delta_odorsdiff, relative_rate_delta_odorsdiff
