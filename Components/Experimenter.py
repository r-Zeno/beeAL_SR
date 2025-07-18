import numpy as np
from pygenn.genn_model import GeNNModel
import time
import os
import json
from helpers import gauss_odor, set_odor_simple

class Experimenter:
    """
    Only runs the predifined experiment using the model defined by the simulator, then saves data after each run.
    - model: the model created with ModelBuilder
    - experiment_parameter: dictionary containing odor/exp parameters, loaded by the simulator from json
    """
    def __init__(self, model, experiment_parameters: dict, sim_path, n_run:int, noise_lvl:float, spk_rec_steps:int, debugmode: bool):

        self.noise_lvl = noise_lvl
        self.model = model
        self.paras = experiment_parameters
        self.rec_states = self.paras["rec_states"]
        self.spk_rec_steps = spk_rec_steps # ugly way to match the spike pulling rate to the model recording steps
        self.n_run = n_run
        self.run_settings= dict()
        self.debug = debugmode

        dir_name = f"run_{str(self.n_run)}"
        self.folder = os.path.join(sim_path, dir_name)
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

    def _noise_lvl_injecter(self):

        corr_noise_lvl = self.noise_lvl/np.sqrt(self.model.dt)
        print(f"injector for run {self.n_run}, applying noise lvl {self.noise_lvl} (corrected: {corr_noise_lvl})")

        for pop in self.paras["noisy_pop"]:
            popobj = self.model.neuron_populations.get(pop)
            print(popobj)
            if popobj is not None:
                print(f"injecting population{popobj}")
                popobj.set_dynamic_param_value("noise_A", corr_noise_lvl)

    def _rec_var_init(self):

        self.pop_to_rec = self.paras["pop_to_rec"]

        spike_t = dict()
        spike_id = dict()
        for pop in self.pop_to_rec:
            spike_t[pop] = []
            spike_id[pop] = []

        print(spike_t)
        print(spike_id)

        # rec var state from neurons
        self.what_to_rec = self.paras["what_to_rec"]
        self.vars_rec = dict()
        for pop, var in self.what_to_rec:
            self.vars_rec[f"{pop}_{var}"] = []

        print(self.vars_rec)
        return spike_t, spike_id

    def _exp_separate_od(self):

        base = np.power(10,0.25)
        c = 12
        on = 1e-7*np.power(base,c)
        off = 0.0
        odor_slot = 0 # more than one odor slot in ors so must specify

        baseline = 1000
        t_relaxation = 1000
        stim_duration = 3000
        t_odor_on = baseline
        t_odor_off = t_odor_on + stim_duration
        sim_time = t_odor_off + t_relaxation

        state_rec_steps = 10 # pull state vars every 10 timesteps (curr every 1 ms)

        var_view = {}
        for pop, var in self.what_to_rec: # getting var directly (more efficient(?))
            var_view[f"{pop}_{var}"] = self.model.neuron_populations[pop].vars[var].view

        ors_population = self.model.neuron_populations["or"]

        print("starting sim...")
        start = time.time()
        i = 0

        for odor in self.odors:
            i += 1

            # init exp params
            int_t = 0
            odor_applied = False
            odor_removed = False
            
            # to load model in a fresh state
            self.model.load(num_recording_timesteps = int(self.paras["spk_rec_steps"]))
            self._noise_lvl_injecter()

            # init spiking data
            spike_t = {}
            spike_id = {}
            for pop in self.pop_to_rec:
                spike_t[pop] = []
                spike_id[pop] = []
            self.vars_rec = {}
            for pop, var in self.what_to_rec:
                self.vars_rec[f"{pop}_{var}"] = []

            f_name = f"odor_{str(i)}"
            exp_s_folder = os.path.join(self.folder, f_name)

            while self.model.t < sim_time:

                if not odor_applied and self.model.t >= t_odor_on:
                    print(f"Time {self.model.t}, applying odor")
                    set_odor_simple(ors_population, odor_slot, odor, on, self.hill_exp)
                    odor_applied = True

                if odor_applied and not odor_removed and self.model.t >= t_odor_off:
                    print(f"Time {self.model.t}, shutting off odor")
                    set_odor_simple(ors_population, odor_slot, odor, off, self.hill_exp)
                    odor_removed = True

                self.model.step_time()
                int_t += 1

                # pulling var states
                if int_t%state_rec_steps == 0:

                    for pop_name, var_name in self.what_to_rec:
                        view_key = f"{pop_name}_{var_name}"

                        var_obj = self.model.neuron_populations[pop_name].vars[var_name]
                        var_obj.pull_from_device()
                        current_var_view = var_view.get(view_key)

                        current_val = np.copy(current_var_view)
                        self.vars_rec[view_key].append(current_val)

                # pulling spikes
                if int_t%self.spk_rec_steps == 0: # every 1000 timesteps (int_t%1000 == 0 checks if int_t/1000 is equal to 0)
                    self.model.pull_recording_buffers_from_device()
                    
                    for pop in self.pop_to_rec:
                        pop_to_pull = self.model.neuron_populations[pop]

                        if (pop_to_pull.spike_recording_data[0][0].size > 0):
                            spike_t[pop].append(pop_to_pull.spike_recording_data[0][0])
                            spike_id[pop].append(pop_to_pull.spike_recording_data[0][1])
                            if self.debug:
                                print(f"spikes fetched for time {self.model.t} from {pop}")
                        else: 
                            if self.debug:
                                print(f"no spikes in t {self.model.t} in {pop}")

            self.model.unload() # clearing model before starting 2nd run

            self._data_saver(spike_t, spike_id, exp_s_folder) # saving data dynamically

        end = time.time()
        timetaken = round(end-start, 2)
        if self.debug:
            print(f"sim ended. it took {timetaken} s.")

    def _exp_consecutive_od(self, spike_t, spike_id):

        exp_folder = self.folder
        base = np.power(10,0.25)
        c = 12
        on = 1e-7*np.power(base,c)
        off = 0.0
        odor_slot = 0

        t_odor1_on = 1000
        t_odor1_off = t_odor1_on + 3000
        t_pause_end = t_odor1_off + 1000
        t_odor2_on = t_pause_end
        t_odor2_off = t_odor2_on + 3000
        sim_time = 8000 # ms

        odor1_applied = False
        odor1_removed = False
        odor2_applied = False
        odor2_removed = False
        
        self.model.load(num_recording_timesteps = int(self.paras["spk_rec_steps"]))
        self._noise_lvl_injecter()

        var_view = {}
        for pop, var in self.what_to_rec: # getting var directly (more efficient(?))
            var_view[f"{pop}_{var}"] = self.model.neuron_populations[pop].vars[var].view

        state_rec_steps = 10 # pull state vars every 10 timesteps (curr every 1 ms)

        ors_population = self.model.neuron_populations["or"]

        # making sure odor is off
        print(f"Initial state: applying 0.0 concentration to type {odor_slot}")
        set_odor_simple(ors_population, odor_slot, self.odors[0], off, self.hill_exp)

        int_t = 0 # init internal counter
        if self.debug:
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

                for pop_name, var_name in self.what_to_rec:
                    view_key = f"{pop_name}_{var_name}"

                    var_obj = self.model.neuron_populations[pop_name].vars[var_name]
                    var_obj.pull_from_device()
                    current_var_view = var_view.get(view_key)

                    current_val = np.copy(current_var_view)
                    self.vars_rec[view_key].append(current_val)

            # pulling spikes
            if int_t%self.spk_rec_steps == 0: # every 1000 timesteps (int_t%1000 == 0 checks if int_t/1000 is equal to 0)
                self.model.pull_recording_buffers_from_device()
                
                for pop in self.pop_to_rec:
                    pop_to_pull = self.model.neuron_populations[pop]

                    if (pop_to_pull.spike_recording_data[0][0].size > 0):
                        spike_t[pop].append(pop_to_pull.spike_recording_data[0][0])
                        spike_id[pop].append(pop_to_pull.spike_recording_data[0][1])
                        print(f"spikes fetched for time {self.model.t} from {pop}")
                    else: print(f"no spikes in t {self.model.t} in {pop}")

        self.model.unload()
                    
        end = time.time()

        timetaken = round(end-start, 2)
        if self.debug:
            print(f"sim ended. it took {timetaken} s.")

        return spike_t, spike_id, exp_folder

    def _data_saver(self, spike_t, spike_id, exp_folder):
        
        os.makedirs(exp_folder, exist_ok=True)
        for pop in self.pop_to_rec:
            if spike_t[pop] is not None:
                spike_t[pop] = np.hstack(spike_t[pop])
                spike_id[pop] = np.hstack(spike_id[pop])

        for pop in self.pop_to_rec:
            np.save(os.path.join(exp_folder, f"{pop}_spike_t.npy"), spike_t[pop])
            np.save(os.path.join(exp_folder, f"{pop}_spike_id.npy"), spike_id[pop])

        if self.rec_states:
            for key, segments_list in self.vars_rec.items():
                if segments_list:
                    self.vars_rec[key] = np.vstack(segments_list)
                else:
                    self.vars_rec[key] = np.array([])

            for pop_var2 in self.vars_rec:
                np.save(os.path.join(exp_folder, f"{pop_var2}_states.npy"), self.vars_rec[pop_var2])
        else: 
            if self.debug: 
                print("Warning: variable states (V) are not being recorded for this run!")

        # saving the run's details into a json
        self.run_settings["noise_lvl"] = self.noise_lvl
        with open(os.path.join(exp_folder, f"run_{self.n_run}_settings.json"), 'w') as fp:
            json.dump(self.run_settings, fp)

    def run(self, exp_1:bool, exp_2:bool):
        """
        starts the experiment
        """
        self._odor_maker()
        spike_t, spike_id = self._rec_var_init()

        if exp_1:
            spike_t_exp, spike_id_exp, exp_folder = self._exp_consecutive_od(spike_t, spike_id)
            self._data_saver(spike_t_exp, spike_id_exp, exp_folder)
        elif exp_2:
            self._exp_separate_od()
        else: raise ValueError("Must choose an experiment in the parameters file!")

        if self.debug:
            print(f"exp run, saved in '{self.folder}'")

        return self.folder
