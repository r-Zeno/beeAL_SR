import numpy as np
import os
from pygenn.genn_model import GeNNModel
from pygenn import CurrentSource


class ExperimentDynamicSingle:

    def __init__(self, paras, model:GeNNModel):

        noise_max = paras["noise"]["noiselvl_min"]
        noise_min = paras["noise"]["noiselvl_max"]
        steps = paras["noise"]["noiselvl_steps"]
        # normalized by integration timestep
        self.noise_lvls = np.divide(np.linspace(noise_min, noise_max, steps), np.sqrt(model.dt))

        self.duration = paras["exp_duration_sec"]
        self.tau = paras["autocorellation_time"]
        self.mean_value = paras["expected_value"]
        self.stim_sd = paras["standard_dev"]

    def _stim_gen(self):


        return stim

    def run(self, run, iteration):

        stim = self._stim_gen()

        # will have to generate an array via currensource on the gpu, since the value is updated every timestep (?)

        tot_time = self.duration * 1000


        return stim, spk_id, spk_t
        
