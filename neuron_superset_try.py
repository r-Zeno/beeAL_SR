import os
import numpy as np

def neuron_spikes_assemble(threshold, paths, paras, pad:bool):

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
                    
                    if t < paras["start_stim"]:

                        if id not in curr_odor_baseline[pop]:
                            curr_odor_baseline[pop][id] = []

                        curr_odor_baseline[pop][id].append(t)

                    elif paras["start_stim"] < t < paras["end_stim"]+padding:

                        if id not in curr_odor_stimulation[pop]:
                            curr_odor_stimulation[pop][id] = []

                        curr_odor_stimulation[pop][id].append(t)
                    else: pass

                final_spk_baseline = {}
                final_spk_stimulation = {}
                for i in range(paras["pop_number"]):
                    final_spk_baseline[i] = curr_odor_baseline[pop].get(i, [])
                    final_spk_stimulation[i] = curr_odor_stimulation[pop].get(i, [])

                curr_odor_baseline = final_spk_baseline
                curr_odor_stimulation = final_spk_stimulation

            curr_run_baseline[odor] = curr_odor_baseline
            curr_run_stimulation[odor] = curr_odor_stimulation

        runs_baseline[run_name] = curr_run_baseline
        runs_stimulation[run_name] = curr_run_stimulation

    spks_split = {}
    spks_split["baseline"] = runs_baseline
    spks_split["stimulation"] = runs_stimulation

def fire_rate(data:dict, paras:dict)
    
    
