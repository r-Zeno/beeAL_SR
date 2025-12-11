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

        self.duration = paras["exp_duration_sec"]
        self.tau = paras["autocorellation_time"]
        self.mean_value = paras["expected_value"]
        self.stim_sd = paras["standard_dev"]

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

    def run(self, run, iteration):

        sim_time = self.duration

        if self.stim_generated is None:
            stim = self._stim_gen(sim_time)
        else: stim = self.stim_generated

        # will have to generate an array via currentsource on the gpu, since the value is updated every timestep (?)




        return stim, spk_id, spk_t
        
