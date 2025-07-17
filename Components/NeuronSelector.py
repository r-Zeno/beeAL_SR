import os
import sys
import numpy as np

class NeuronSelector:

    def __init__(self, paths, paras, only0noise:bool, pad:bool):

        self.paths = paths
        self.paras = paras
        self.pad = pad
        self.only0noise = only0noise

    def _neuron_spikes_assemble(self):

        if self.pad:
            padding = 200
        else: padding = 0

        runs_baseline = {}
        runs_stimulation = {}
        for run in self.paths:

            run_name = os.path.basename(run)
            curr_run_baseline = {}
            curr_run_stimulation = {}

            for odor in self.paras["odors"]:

                curr_odor_baseline = {}
                curr_odor_stimulation = {}

                for pop in self.paras["which_pop"]: # useless for now since only 1 pop, included for scalability

                    curr_spike_ts = np.load(os.path.join(run, odor, f"{pop}_spike_t.npy"))
                    curr_spike_ids = np.load(os.path.join(run, odor, f"{pop}_spike_id.npy"))
                    curr_pop_idxt = np.stack((curr_spike_ids, curr_spike_ts), 1)
                    curr_odor_baseline[pop] = {}
                    curr_odor_stimulation[pop] = {}

                    for (id, t) in curr_pop_idxt:
                        key = int(id)
                        if t < self.paras["start_stim"]:

                            if key not in curr_odor_baseline[pop]:
                                curr_odor_baseline[pop][key] = []

                            curr_odor_baseline[pop][key].append(t)

                        elif self.paras["start_stim"] < t < self.paras["end_stim"]+padding:

                            if key not in curr_odor_stimulation[pop]:
                                curr_odor_stimulation[pop][key] = []

                            curr_odor_stimulation[pop][key].append(t)
                        else: pass

                    final_spk_baseline = {}
                    final_spk_stimulation = {}
                    for i in range(self.paras["pop_number"]):
                        final_spk_baseline[i] = curr_odor_baseline[pop].get(i, [])
                        final_spk_stimulation[i] = curr_odor_stimulation[pop].get(i, [])

                    curr_odor_baseline[pop] = final_spk_baseline
                    curr_odor_stimulation[pop] = final_spk_stimulation

                curr_run_baseline[odor] = curr_odor_baseline
                curr_run_stimulation[odor] = curr_odor_stimulation

            runs_baseline[run_name] = curr_run_baseline
            runs_stimulation[run_name] = curr_run_stimulation

        spks_split = {}
        spks_split["baseline"] = runs_baseline
        spks_split["stimulation"] = runs_stimulation

        return spks_split

    def _fire_rate(self, data:dict):
        
        baseline_t = self.paras["start_stim"] / 1000.0 # from ms to s (rate in Hz)
        stimulation_t = (self.paras["end_stim"] - self.paras["start_stim"]) / 1000.0
        print(f"DEBUG: Baseline Duration = {baseline_t}s")
        print(f"DEBUG: Stimulation Duration = {stimulation_t}s")
        rates = {}
        for state, runs in data.items():

            rates[state] = {}

            if state == "baseline":
                duration = baseline_t
            else: duration = stimulation_t

            for run_n, odors in runs.items():
                rates[state][run_n] = {}

                for odor_n, pops in odors.items():
                    rates[state][run_n][odor_n] = {}

                    for pop_n, neurons in pops.items():
                        rates[state][run_n][odor_n][pop_n] = {}

                        for neuron, spikes in neurons.items():

                            curr_rate = len(spikes) / duration
                            rates[state][run_n][odor_n][pop_n][neuron] = curr_rate
        
        return rates

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

    def _select_neurons_0noise(self, rates:dict):

        n_neurons = self.paras["pop_number"]
        odors = self.paras["odors"]
        pops = self.paras["which_pop"]
        run_idx = "run_0" # assuming that the first run is at 0 noise

        decision_vector = np.zeros(n_neurons, dtype=bool)
        
        for odor in odors:
            for pop in pops:
                for neuron in range(n_neurons):

                    curr_baseline_rate = rates["baseline"][run_idx][odor][pop][neuron]
                    curr_stimulation_rate = rates["stimulation"][run_idx][odor][pop][neuron]

                    if curr_stimulation_rate > (curr_baseline_rate + self.paras["threshold_0noise"]): # at 0 noise only change if touched by odor

                        decision_vector[neuron] = True

        responsive_ns = np.where(decision_vector)[0]
        neurons2analyze = responsive_ns.tolist()

        return neurons2analyze, decision_vector
    
    def select(self):

        spks_split = self._neuron_spikes_assemble()
        rates = self._fire_rate(spks_split)

        if self.only0noise:
            neurons2analyze, decision_matrix = self._select_neurons_0noise(rates)
        else: neurons2analyze, decision_matrix = self._select_neurons(rates)

        return neurons2analyze, decision_matrix