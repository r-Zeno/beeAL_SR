import numpy as np
import os
from pygenn.genn_model import GeNNModel
from pygenn import genn_model


class ExperimentDynamicSingle:

    def __init__(self, paras, model:GeNNModel, stim_generated, num_orn, num_pn, debug:bool):

        self.debug = debug
        self.stim_generated = stim_generated
        noise_min = paras["noise"]["noiselvl_min"]
        noise_max = paras["noise"]["noiselvl_max"]
        steps = paras["noise"]["noiselvl_steps"]
        self.noisy_pop = paras["noise"]["noisy_pop"]
        # normalized by integration timestep
        self.noise_lvls = np.divide(np.linspace(noise_min, noise_max, steps), np.sqrt(model.dt))
        self.model = model
        self.num_orn = num_orn
        self.num_pn = num_pn

        self.spk_rec_steps = paras["spike_recording_steps_ms"]
        self.pop2record = paras["pop_to_record"]

        self.duration = paras["exp_duration_s"]
        self.tau = paras["stimulus"]["autocorrelation_time_s"]
        self.mean_value = paras["stimulus"]["expected_value_nA"]
        self.stim_sd = paras["stimulus"]["standard_dev_nA"]
        self.seed = paras["stimulus"]["random_seed"]
        if self.seed is False:
            self.seed = None
        self.dt = model.dt

        self.orn_to_pn_input_scale = paras["orn_n"]["pn_input_scale"]

        self.spike_t = {}
        self.spike_id = {}
        for pop in self.pop2record:
            self.spike_t[pop] = []
            self.spike_id[pop] = []


    def _stim_gen(self, time, dt, tau_s, mean, sigma, seed=None):
        """
        Generates an aperiodic gaussian signal following the methods of Duan et al. (2009)
        """
        if seed is not None:
            np.random.seed(seed)
        
        steps = int(time/dt)
        tau_ms = tau_s * 1000.0

        time = np.linspace(0, time, steps)
        stim = np.zeros(steps, dtype=np.float32)
        stim[0] = mean
        noise_var = sigma**2
        noise_scaling = np.sqrt((2 * noise_var * dt)/tau_ms)
        noise = np.random.normal(0, 1, steps)

        for i in range(1, steps):
            drift = -(1/tau_ms) * (stim[i-1] - mean) * dt
            diffusion = noise_scaling * noise[i-1]
            stim[i] = stim[i-1] + drift + diffusion

        return stim
    
    def _noise_lvl_injecter(self, run):

        noise_lvl = self.noise_lvls[run]

        for pop in self.noisy_pop:
            popobj = self.model.neuron_populations.get(pop)

            if popobj is not None:
                print(f"applying noise lvl {noise_lvl} in pop {pop}...")
                popobj.set_dynamic_param_value("noise_A", noise_lvl)

    def _input_orn_scale(self):

        ratio = int(self.num_orn/self.num_pn)
        dist = ratio/120
        if dist is not 1:
            scaling = np.float32(1/dist)
        else: scaling = np.float32(1)

        popobj = self.model.neuron_populations.get("pn")
        popobj.set_dynamic_param_value("r_scale", scaling)
        if self.debug: print(f"applied scaling {scaling} to pn input")

    def run(self, run):

        sim_time = self.duration*1000 # in ms
        pull_rate_steps = int(self.spk_rec_steps / self.model.dt)

        if self.stim_generated is None:
            stim = self._stim_gen(sim_time, self.dt, self.tau, self.mean_value, self.stim_sd, self.seed)
        else: stim = self.stim_generated

        self.model.load(num_recording_timesteps = pull_rate_steps)

        self._noise_lvl_injecter(run)

        if self.orn_to_pn_input_scale:
            self._input_orn_scale()
        
        var2input = self.model.input_model.extra_global_params["input_stream"]
        var2input.view[:] = stim
        var2input.push_to_device()

        # stab_time_dt = 0.2*1000/self.model.dt
        while self.model.t <= sim_time:
            # need to give short time to stabilize LIF (stab time)        

            self.model.step_time()

            # spike pulling loop
            if self.model.timestep%pull_rate_steps == 0:
                
                self.model.pull_recording_buffers_from_device()

                for pop in self.pop2record:
                    pop2pull = self.model.neuron_populations[pop]

                    if pop2pull.spike_recording_data[0][0].size > 0:
                        self.spike_t[pop].append(pop2pull.spike_recording_data[0][0])
                        self.spike_id[pop].append(pop2pull.spike_recording_data[0][1])
                        if self.debug: print(f"spikes fetched for time {self.model.t} in {pop}")
                    else: 
                        if self.debug: print(f"no spikes at time {self.model.t} in {pop}")

        self.model.unload()

        return stim, self.spike_id, self.spike_t
