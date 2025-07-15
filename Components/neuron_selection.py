import os
import numpy as np

def neuron_spikes_assemble(paths, paras, pad:bool):

    if pad:
        padding = 200
    else: padding = 0

    runs_baseline = {}
    runs_stimulation = {}
    for run in paths:

        run_name = os.path.basename(run)
        curr_run_baseline = {}
        curr_run_stimulation = {}

        for odor in paras["odors"]:

            curr_odor_baseline = {}
            curr_odor_stimulation = {}

            for pop in paras["which_pop"]: # useless for now since only 1 pop, included for scalability

                curr_spike_ts = np.load(os.path.join(run, odor, f"{pop}_spike_t.npy"))
                curr_spike_ids = np.load(os.path.join(run, odor, f"{pop}_spike_id.npy"))
                curr_pop_idxt = np.stack((curr_spike_ids, curr_spike_ts), 1)
                curr_odor_baseline[pop] = {}
                curr_odor_stimulation[pop] = {}

                for (id, t) in curr_pop_idxt:
                    key = int(id)
                    if t < paras["start_stim"]:

                        if key not in curr_odor_baseline[pop]:
                            curr_odor_baseline[pop][key] = []

                        curr_odor_baseline[pop][key].append(t)

                    elif paras["start_stim"] < t < paras["end_stim"]+padding:

                        if key not in curr_odor_stimulation[pop]:
                            curr_odor_stimulation[pop][key] = []

                        curr_odor_stimulation[pop][key].append(t)
                    else: pass

                final_spk_baseline = {}
                final_spk_stimulation = {}
                for i in range(paras["pop_number"]):
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

def fire_rate(data:dict, paras:dict):
    
    baseline_t = paras["start_stim"] / 1000.0 # from ms to s (rate in Hz)
    stimulation_t = (paras["end_stim"] - paras["start_stim"]) / 1000.0

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

def select_neurons(rates:dict, paras:dict):

    runs = list(rates["baseline"].keys())
    n_runs = len(runs)

    n_neurons = paras["pop_number"]
    odors = paras["odors"]
    pops = paras["which_pop"]

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

                    if curr_stimulation_rate > (curr_baseline_rate + paras["threshold"]):

                        idx = run_idx[run]
                        decision_matrix[neuron, idx] = True

    responsive_ns = np.where(np.any(decision_matrix, axis=1))[0]
    neurons2analyze = responsive_ns.tolist()

    return neurons2analyze, decision_matrix
