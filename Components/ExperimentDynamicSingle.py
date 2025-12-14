import numpy as np
import os
from pygenn.genn_model import GeNNModel
from pygenn import genn_model


class ExperimentDynamicSingle:

    def __init__(self, paras, model:GeNNModel, stim_generated, debug:bool):

        self.debug = debug
        self.stim_generated = stim_generated
        noise_max = paras["noise"]["noiselvl_min"]
        noise_min = paras["noise"]["noiselvl_max"]
        steps = paras["noise"]["noiselvl_steps"]
        self.noisy_pop = paras["noise"]["noisy_pop"]
        # normalized by integration timestep
        self.noise_lvls = np.divide(np.linspace(noise_min, noise_max, steps), np.sqrt(model.dt))
        self.model = model

        self.spk_rec_steps = paras["spk_rec_steps_ms"]
        self.pop2record = paras["pop_to_record"]

        self.duration = paras["exp_duration_s"]
        self.tau = paras["autocorellation_time_s"]
        self.mean_value = paras["expected_value"]
        self.stim_sd = paras["standard_dev"]
        self.seed = paras["rndm_seed"]
        if self.seed == 0:
            self.seed = None
        self.dt = model.dt

        self.spike_t = {}
        self.spike_id = {}
        for pop in self.pop2record:
            self.spike_t[pop] = np.array()
            self.spike_id[pop] = np.array()


    def _stim_gen(self, time, dt, tau_s, mean, sigma, seed=None):
        """
        Generates an aperiodic gaussian signal following the methods of Duan et al. (2009)
        """
        if seed is not None:
            np.random.seed(seed)
        
        steps = int(time/dt)

        time = np.linspace(0, time, steps)
        stim = np.zeros(steps)
        stim[0] = mean
        noise_scaling = np.sqrt((2 * sigma * dt)/tau_s)
        noise = np.random.normal(0, 1, steps)

        for i in range(1, steps):
            drift = -(1/tau_s) * (stim[i-1] - mean) * dt
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

    def run(self, run):

        sim_time = self.duration * 1000 # in ms

        if self.stim_generated is None:
            stim = self._stim_gen(sim_time, self.dt, self.tau, self.mean_value, self.stim_sd, self.seed)
        else: stim = self.stim_generated

        self.model.load(num_recording_timesteps = int(self.spk_rec_steps))

        self._noise_lvl_injecter(run)
        
        # if this returns error, then the custom current source model cannot be added to the model object
        # and will need to be passed by Simulator from ModelBuilder :(
        var2input = self.model.input_source.extra_global_params["input_stream"]
        var2input.view[:] = stim
        var2input.push_to_device()

        while self.model.t <= sim_time:
            # need to give short time to stabilize LIF?            

            self.model.step_time()

            # spike train pulling loop
            if self.model.t%self.spk_rec_steps:

                for pop in self.pop2record:
                    pop2pull = self.model.neuron_populations[pop]

                    if pop2pull.spike_recording_data[0][0] > 0:
                        self.spike_t[pop].append(pop2pull.spike_recording_data[0][0])
                        self.spike_id[pop].append(pop2pull.spike_recording_data[0][1])
                        if self.debug: print(f"spikes fetched for time {self.model.t} in {pop}")
                    else: 
                        if self.debug: print(f"no spikes at time {self.model.t} in {pop}")

        self.model.unload()

        return stim, self.spike_id, self.spike_t
