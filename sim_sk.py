import numpy as np
from matplotlib import pyplot as plt
import pygenn
from pygenn import create_var_ref, init_postsynaptic, init_sparse_connectivity
from pygenn.genn_model import GeNNModel, create_weight_update_model, create_postsynaptic_model, create_sparse_connect_init_snippet, create_var_init_snippet, init_weight_update
import time
import os
from model_builder import model_builder
from protocol_builder import protocol_builder
try:
    import GPUtil
except: print("GPUtil not installed, you will have no info on gpu status, sim will start anyway.")

class simulator:
    def __init__(self, noise, paras, dt = 0.1, eLns = False):
        self.eLns = eLns
        self.paras = paras

        model = model_builder(paras)
        protocol = protocol_builder(paras)

    while model.t < sim_time:

        if not odor1_applied and model.t >= t_odor1_on:
            print(f"Time {model.t}, applying odor 1")
            set_odor_simple(ors_population, odor_slot, odors[0], on, hill_exp)
            odor1_applied = True

        if odor1_applied and not odor1_removed and model.t >= t_odor1_off:
            print(f"Time {model.t}, shutting off odor 1")
            set_odor_simple(ors_population, odor_slot, odors[0], off, hill_exp)
            odor1_removed = True

        if odor1_removed and not odor2_applied and model.t >= t_odor2_on:
            print(f"Time {model.t}, applying odor 2")
            set_odor_simple(ors_population, odor_slot, odors[1], on, hill_exp)
            odor2_applied = True

        if odor2_applied and not odor2_removed and model.t >= t_odor2_off:
            print(f"Time {model.t}, shutting off odor 2")
            set_odor_simple(ors_population, odor_slot, odors[1], off, hill_exp)
            odor2_removed = True

        model.step_time()
        int_t += 1

        # pulling var states
        if int_t%state_rec_steps == 0:

            for pop_name, var_name in what_to_rec:
                view_key = f"{pop_name}_{var_name}"

                var_obj = model.neuron_populations[pop_name].vars[var_name]
                var_obj.pull_from_device()
                current_var_view = var_view.get(view_key)

                current_val = np.copy(current_var_view)
                vars_rec[view_key].append(current_val)

        # pulling spikes
        if int_t%spk_rec_steps == 0: # every 1000 timesteps (int_t%1000 == 0 checks if int_t/1000 is equal to 0)
            model.pull_recording_buffers_from_device()
            
            for pop in pop_to_rec:
                pop_to_pull = model.neuron_populations[pop]

                if (pop_to_pull.spike_recording_data[0][0].size > 0):
                    spike_t[pop].append(pop_to_pull.spike_recording_data[0][0])
                    spike_id[pop].append(pop_to_pull.spike_recording_data[0][1])
                    print(f"spikes fetched for time {model.t} from {pop}")
                else: print(f"no spikes in t {model.t} in {pop}")
                
    end = time.time()

    timetaken = round(end-start, 2)
    print(f"sim ended. it took {timetaken} s.")




        

