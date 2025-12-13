import os
import numpy as np

class Analyzer:

    def __init__(self, paras, stim_path, spk_t_paths, spk_id_paths):

        self.exp_type = paras["which_exp"]
        self.stim_path = stim_path
        self.spk_t_paths = spk_t_paths
        self.spk_id_paths = spk_id_paths
        self.runs = paras["num_runs"]
        self.trials = paras["num_trials"]

        analysis_paras = paras["analysis_parameters"]

    def run(self):

        match self.exp_type:
            
            case "static_single":

                # to move everything here
                raise ValueError("not ready yet")
            
            case "DynamicSingle":

                mi_val_path = []

                for 