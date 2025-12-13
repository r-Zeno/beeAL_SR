import os
import numpy as np
from helpers import data_log_compile
from mi_analysis_dynamic_single import *

class Analyzer:

    def __init__(self, paras, stim_path, data_paths):

        self.exp_type = paras["which_exp"]
        self.stim_path = stim_path
        self.runs = paras["num_runs"]
        self.trials = paras["num_trials"]
        self.data = data_log_compile(data_paths)

        analysis_paras = paras["analysis_parameters"]

    def run(self):

        match self.exp_type:
            
            case "static_single":

                # to move everything here
                raise ValueError("not ready yet")
            
            case "DynamicSingle":

                mi_vals = []
                stim = np.load(self.stim_path)

                sorted_lvls = sorted(self.data.keys())
                for lvl in sorted_lvls:

                    sorted_trials = sorted(self.data[lvl].keys())
                    for it in sorted_trials:

                        for pop in self.pop2analyze:

                            pop_data = self.data[lvl][it][pop]

                            spk_id = np.load(pop_data["spk_id_path"])
                            spk_t = np.load(pop_data["spk_t_path"])

                            mi = mi_analysis_dynamic_single(stim, spk_id, spk_t)
                            mi_vals.append(mi)

                np.save(mi_vals, )
