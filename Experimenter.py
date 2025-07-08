import numpy as np
from pygenn.genn_model import GeNNModel
import time
import os
from helpers import gauss_odor, set_odor_simple

class Experimenter:
    """
    Only runs the predifined experiment using the model defined by the simulator, then saves data after each run.
    - model: the model created with ModelBuilder
    - experiment_parameter: dictionary containing odor/exp parameters, loaded by the simulator from json
    """
    def __init__(self, model, experiment_parameters: dict):

        self.model = model
        self.paras = experiment_parameters
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        now = time.strftime("%Y%m%d_%H%M%S")
        dirname = f"sim_{now}"
        self.folder = os.path.join(current_dir, dirname)
        os.makedirs(self.folder)

    def _odor_maker(self):
        
        self.odors= []
        odor_stable_params = self.paras["odors_stable_params"]
        
        od1_m = self.paras["odor1_midpoint"]
        od1_sd_b = self.paras["odor1_sd"]
        od1 = gauss_odor(n_glo=self.paras["num_glo"], m = od1_m, sd_b= od1_sd_b, a_rate= odor_stable_params["a_rate"], A= odor_stable_params["A"])
        self.odors.append(np.copy(od1))

        od2_m = self.paras["odor2_midpoint"]
        od2_sd_b = self.paras["odor2_sd"]
        od2 = gauss_odor(n_glo=self.paras["num_glo"], m = od2_m, sd_b= od2_sd_b, a_rate= odor_stable_params["a_rate"], A= odor_stable_params["A"])
        self.odors.append(np.copy(od2))

        # defining Hill coefficient
        self.hill_exp= np.random.uniform(0.95, 1.05, self.paras["num_glo"])
        np.save(os.path.join(self.folder,"_hill"), self.hill_exp)

    def _experiment_runner(self):

        base = np.power(10,0.25)
        c = 12
        on = 1e-7*np.power(base,c)
        off = 0.0
        odor_slot = 0

        t_baseline_end = 1000
        t_odor1_on = 1000
        t_odor1_off = t_odor1_on + 3000
        t_pause_end = t_odor1_off + 1000
        t_odor2_on = t_pause_end
        t_odor2_off = t_odor2_on + 3000

        odor1_applied = False
        odor1_removed = False
        odor2_applied = False
        odor2_removed = False

        ## rec spikes from neurons
        self.pop_to_rec = self.paras["pop_to_rec"]
        sim_time = 8000 # ms

        self.spike_t = dict()
        self.spike_id = dict()
        for pop in self.pop_to_rec:
            self.spike_t[pop] = []
            self.spike_id[pop] = []

        print(self.spike_t)
        print(self.spike_id)

        # rec var state from neurons
        what_to_rec = self.paras["what_to_rec"]
        self.vars_rec = dict()
        for pop, var in what_to_rec:
            self.vars_rec[f"{pop}_{var}"] = []

        print(self.vars_rec)

        var_view = {}
        for pop, var in what_to_rec: # getting var directly (more efficient(?))
            var_view[f"{pop}_{var}"] = self.model.neuron_populations[pop].vars[var].view

        state_rec_steps = 10 # pull state vars every 10 timesteps (curr every 1 ms)

        ## Preparing protocol
        ors_population = self.model.neuron_populations["or"]

        # making sure odor is off
        print(f"Initial state: applying 0.0 concentration to type {odor_slot}")
        set_odor_simple(ors_population, odor_slot, self.odors[0], off, self.hill_exp)


        ## start simulation
        int_t = 0 # init internal counter
        print("starting sim...")
        start = time.time()

        while self.model.t < sim_time:

            if not odor1_applied and self.model.t >= t_odor1_on:
                print(f"Time {self.model.t}, applying odor 1")
                set_odor_simple(ors_population, odor_slot, self.odors[0], on, self.hill_exp)
                odor1_applied = True

            if odor1_applied and not odor1_removed and self.model.t >= t_odor1_off:
                print(f"Time {self.model.t}, shutting off odor 1")
                set_odor_simple(ors_population, odor_slot, self.odors[0], off, self.hill_exp)
                odor1_removed = True

            if odor1_removed and not odor2_applied and self.model.t >= t_odor2_on:
                print(f"Time {self.model.t}, applying odor 2")
                set_odor_simple(ors_population, odor_slot, self.odors[1], on, self.hill_exp)
                odor2_applied = True

            if odor2_applied and not odor2_removed and self.model.t >= t_odor2_off:
                print(f"Time {self.model.t}, shutting off odor 2")
                set_odor_simple(ors_population, odor_slot, self.odors[1], off, self.hill_exp)
                odor2_removed = True

            self.model.step_time()
            int_t += 1

            # pulling var states
            if int_t%state_rec_steps == 0:

                for pop_name, var_name in what_to_rec:
                    view_key = f"{pop_name}_{var_name}"

                    var_obj = self.model.neuron_populations[pop_name].vars[var_name]
                    var_obj.pull_from_device()
                    current_var_view = var_view.get(view_key)

                    current_val = np.copy(current_var_view)
                    self.vars_rec[view_key].append(current_val)

            # pulling spikes
            if int_t%1000 == 0: # every 1000 timesteps (int_t%1000 == 0 checks if int_t/1000 is equal to 0)
                self.model.pull_recording_buffers_from_device()
                
                for pop in self.pop_to_rec:
                    pop_to_pull = self.model.neuron_populations[pop]

                    if (pop_to_pull.spike_recording_data[0][0].size > 0):
                        self.spike_t[pop].append(pop_to_pull.spike_recording_data[0][0])
                        self.spike_id[pop].append(pop_to_pull.spike_recording_data[0][1])
                        print(f"spikes fetched for time {self.model.t} from {pop}")
                    else: print(f"no spikes in t {self.model.t} in {pop}")
                    
        end = time.time()

        timetaken = round(end-start, 2)
        print(f"sim ended. it took {timetaken} s.")

    def _data_saver(self):

        for pop in self.pop_to_rec:
            if self.spike_t[pop]:
                self.spike_t[pop] = np.hstack(self.spike_t[pop])
                self.spike_id[pop] = np.hstack(self.spike_id[pop])

        for pop in self.pop_to_rec:
            np.save(os.path.join(self.folder, pop + "_spike_t.npy"), self.spike_t[pop])
            np.save(os.path.join(self.folder, pop + "_spike_id.npy"), self.spike_id[pop])

        final_recorded_vars = {}
        for key, segments_list in self.vars_rec.items():
            if segments_list:
                self.vars_rec[key] = np.vstack(segments_list)
            else:
                self.vars_rec[key] = np.array([])

        for pop_var2 in self.vars_rec:
            np.save(os.path.join(self.folder, f"{pop_var2}_states.npy"), self.vars_rec[pop_var2])

    def run(self):
        """
        starts the experiment
        """
        self._odor_maker()
        self._experiment_runner()
        self._data_saver()
        print(f"exp run, saved in '{self.folder}'")

        return self.folder
