import numpy as np
import os
from pygenn.genn_model import GeNNModel


class ExperimentDynamicSingle:

    def __init__(self, paras, model:GeNNModel):

        noise_max = paras["noise"]["noiselvl_min"]
        noise_min = paras["noise"]["noiselvl_max"]
        steps = paras["noise"]["noiselvl_steps"]
        # normalized by integration timestep
        self.noise_lvls = np.divide(np.linspace(noise_min, noise_max, steps), np.sqrt(model.dt))

    def _stim_gen(self):



        return stim

    def run(self, run, iteration):


        return stim, spk_id, spk_t
        
