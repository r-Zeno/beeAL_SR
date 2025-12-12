import numpy as np
import os
from pygenn.genn_model import GeNNModel
from pygenn import CurrentSource


class ExperimentDynamicSingle:

    def __init__(self, paras, model:GeNNModel, stim_generated):

        self.stim_generated = stim_generated
        noise_max = paras["noise"]["noiselvl_min"]
        noise_min = paras["noise"]["noiselvl_max"]
        steps = paras["noise"]["noiselvl_steps"]
        # normalized by integration timestep
        self.noise_lvls = np.divide(np.linspace(noise_min, noise_max, steps), np.sqrt(model.dt))
        self.model = model

        self.duration = paras["exp_duration_s"]
        self.tau = paras["autocorellation_time_s"]
        self.mean_value = paras["expected_value"]
        self.stim_sd = paras["standard_dev"]
        self.seed = paras["rndm_seed"]
        if self.seed == 0:
            self.seed = None
        self.dt = model.dt

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

        #if self.debug:
        #    plot

        return stim

    def run(self, run, iteration):

        sim_time = self.duration

        if self.stim_generated is None:
            stim = self._stim_gen(sim_time, self.dt, self.tau, self.mean_value, self.stim_sd, self.seed)
        else: stim = self.stim_generated

        # will have to generate an array via currentsource on the gpu, since the value is updated every timestep (?)
        # should stimulus be inputted to ors or orns directly?

        while self.model.t < sim_time:
            # need to give short time to stabilize to LIF?




        return stim, spk_id, spk_t
        
