import os
import numpy as np
from scipy import stats as st
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

                stim = np.load(self.stim_path)

                sorted_lvls = sorted(self.data.keys())

                ncols = max(self.data.keys()) + 1
                nrows = max(self.data[1].keys()) + 1
                mi_vals = np.zeros((nrows, ncols))
                corr_vals = np.zeros((nrows, ncols))

                for lvl in sorted_lvls:

                    sorted_trials = sorted(self.data[lvl].keys())
                    for it in sorted_trials:

                        for pop in self.pop2analyze:

                            pop_data = self.data[lvl][it][pop]

                            spk_id = np.load(pop_data["spk_id_path"])
                            spk_t = np.load(pop_data["spk_t_path"])

                            mi, smooth_rate = mi_analysis_dynamic_single(stim, spk_id, spk_t, self.paras, self.debugmode, self.folder, lvl, it)
                            mi_vals[it][lvl] = mi

                            stim_z = st.zscore(stim, axis=None)
                            smooth_rate_z = st.zscore(smooth_rate, axis=None)
                            corr = np.corrcoef(stim_z, smooth_rate_z)
                            corr_vals[it][lvl] = corr[0][1]


                mean_mi = np.mean(mi_vals, axis=0)
                sd_mi = np.std(mi_vals, axis=0)
                mean_corr = np.mean(corr_vals, axis=0)
                sd_corr = np.std(corr_vals, axis=0)

                mi_path = os.path.join(self.folder, "mi_values.npy")
                mean_mi_path = os.path.join(self.folder, "mean_mi.npy")
                sd_mi_path = os.path.join(self.folder, "sd_mi.npy")
                mean_corr_path = os.path.join(self.folder, "mean_corr.npy")
                sd_corr_path = os.path.join(self.folder, "sd_corr.npy")

                np.save(mi_path, mi_vals), np.save(mean_mi_path, mean_mi), np.save(sd_mi_path, sd_mi)
                np.save(mean_corr_path, mean_corr), np.save(sd_corr_path, sd_corr)

                if self.debugmode: print(f"mean mi values: {mean_mi}, with sd: {sd_mi}")
                if self.debugmode: print(f"mean corr values: {mean_corr}, with sd: {sd_corr}")

                return mi_path
