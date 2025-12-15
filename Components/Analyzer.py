import os
import numpy as np
from helpers import data_log_compile
from mi_analysis_dynamic_single import mi_analysis_dynamic_single

class Analyzer:

    def __init__(self, folder, paras, stim_path, data_paths, which_exp, debugmode:bool):

        self.paras = paras
        self.exp_type = which_exp
        self.stim_path = stim_path
        self.data = data_log_compile(data_paths)
        self.folder = folder
        self.pop2analyze = self.paras["pop_to_analyze"]
        self.debugmode = debugmode

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

                            mi = mi_analysis_dynamic_single(stim, spk_id, spk_t, self.paras)
                            mi_vals.append(mi)

                mi_vals_flat = np.array(mi_vals)
                mi_path = os.path.join(self.folder, "mi_values.npy")
                np.save(mi_path, mi_vals_flat)

                if self.debugmode: print(list(mi_vals))

                return mi_path
