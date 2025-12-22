import os
import numpy as np
import time
from joblib import Parallel, delayed
from scipy import stats as st
from helpers import data_log_compile
from mi_analysis_dynamic_single import mi_analysis_dynamic_single

class Analyzer:

    def __init__(self, folder, paras_an, paras_model, stim_path, data_paths, which_exp, pop_numbers, debugmode:bool):

        self.paras_an = paras_an
        self.paras_model = paras_model
        self.exp_type = which_exp
        self.stim_path = stim_path
        self.data = data_log_compile(data_paths)
        self.folder = folder
        self.pop2analyze = self.paras_an["pop_to_analyze"]
        self.pop_numbers = pop_numbers
        self.debugmode = debugmode

    def run(self):

        match self.exp_type:
            
            case "static_single":

                # to move everything here
                raise ValueError("not ready yet")
            
            case "DynamicSingle":

                if self.debugmode:
                    start_an = time.perf_counter()
                    print("Starting analysis...")

                stim = np.load(self.stim_path, mmap_mode='r')

                sorted_nums = sorted(self.data.keys())

                first_key = next(iter(self.data))
                # because the first key can be any number, cannot be hardcoded
                ncols = max(self.data[first_key].keys()) + 1
                nrows = max(self.data[first_key][0].keys()) + 1

                mi_vals = {}
                mi_bin_vals = {}
                corr_vals = {}

                tasks = []
                for num in sorted_nums:
                    
                    mi_vals[num] = np.zeros((nrows, ncols))
                    mi_bin_vals[num] = np.zeros((nrows, ncols))
                    corr_vals[num] = np.zeros((nrows, ncols))

                    sorted_lvls = sorted(self.data[num].keys())
                    for lvl in sorted_lvls:
                            
                        sorted_trials = sorted(self.data[num][lvl].keys())
                        for it in sorted_trials:
                            tasks.append(
                                delayed(analysis_trial_dynamic_single)(
                                    num,
                                    lvl,
                                    it,
                                    self.pop2analyze,
                                    self.data[num][lvl][it],
                                    stim,
                                    self.paras_an,
                                    self.paras_model,
                                    self.folder,
                                    self.debugmode
                                )
                            )

                results = Parallel(n_jobs = -1, verbose = 5, max_nbytes = None)(tasks)

                for res in results:
                    num_w, lvl_w, it_w, mi_w, mi_bin_w, corr_w = res
                    mi_vals[num_w][it_w][lvl_w] = mi_w
                    mi_bin_vals[num_w][it_w][lvl_w] = mi_bin_w
                    corr_vals[num_w][it_w][lvl_w] = corr_w

                mean_mi_paths = []
                mean_mi_bin_paths = []
                for n in self.pop_numbers:
                    n = int(n)
                    mean_mi = np.mean(mi_vals[n], axis=0)
                    sd_mi = np.std(mi_vals[n], axis=0)
                    mean_mi_bin = np.mean(mi_bin_vals[n], axis=0)
                    sd_mi_bin = np.std(mi_bin_vals[n], axis=0)
                    mean_corr = np.mean(corr_vals[n], axis=0)
                    sd_corr = np.std(corr_vals[n], axis=0)

                    mi_path = os.path.join(self.folder, f"mi_values_pop{n}.npy")
                    mean_mi_path = os.path.join(self.folder, f"mean_mi_pop{n}.npy")
                    sd_mi_path = os.path.join(self.folder, f"sd_mi_pop{n}.npy")
                    mi_bin_path = os.path.join(self.folder, f"mi_bin_values_pop{n}.npy")
                    mean_mi_bin_path = os.path.join(self.folder, f"mean_mi_bin_pop{n}.npy")
                    sd_mi_bin_path = os.path.join(self.folder, f"sd_mi_bin_pop{n}.npy")
                    mean_corr_path = os.path.join(self.folder, f"mean_corr_pop{n}.npy")
                    sd_corr_path = os.path.join(self.folder, f"sd_corr_pop{n}.npy")

                    mean_mi_paths.append(mean_mi_path)
                    mean_mi_bin_paths.append(mean_mi_bin_path)

                    np.save(mi_path, mi_vals), np.save(mean_mi_path, mean_mi), np.save(sd_mi_path, sd_mi)
                    np.save(mi_bin_path, mi_bin_vals), np.save(mean_mi_bin_path, mean_mi_bin), np.save(sd_mi_bin_path, sd_mi_bin)
                    np.save(mean_corr_path, mean_corr), np.save(sd_corr_path, sd_corr)

                max_mis = np.zeros(np.size(self.pop_numbers))
                max_mis_bin = np.zeros(np.size(self.pop_numbers))
                for i in range(len(mean_mi_paths)):
                    mean_mi_curr = np.load(mean_mi_paths[i])
                    mean_mi_bin_curr = np.load(mean_mi_bin_paths[i])

                    max_mis[i] = np.max(mean_mi_curr)
                    max_mis_bin[i] = np.max(mean_mi_bin_curr)

                max_mis_path = os.path.join(self.folder, "max_mi_valuexpop.npy")
                max_mis_bin_path = os.path.join(self.folder, "max_mi_bin_valuexpop.npy")
                np.save(max_mis_path, max_mis)
                np.save(max_mis_bin_path, max_mis_bin)

                if self.debugmode:
                    end_an = time.perf_counter()
                    print(f"analysis ended, it took {end_an-start_an:4f} secs, or {(end_an-start_an)/60:4f} mins")
                if self.debugmode: print(f"mean mi values: {mean_mi}, with sd: {sd_mi}")
                if self.debugmode: print(f"mean mi bin values: {mean_mi_bin}, with sd: {sd_mi_bin}")
                if self.debugmode: print(f"mean corr values: {mean_corr}, with sd: {sd_corr}")

                return mi_path

def analysis_trial_dynamic_single(num, lvl, it, pops, data, stim, paras_an, paras_model, folder, debugmode):
    """
    Function that defines the mi/corr computation for each worker to allow parallelization.
    """
    mi_workr = 0.0
    corr_workr = 0.0

    for pop in pops:

        pop_data = data[pop]

        spk_id = np.load(pop_data["spk_id_path"])
        spk_t = np.load(pop_data["spk_t_path"])

        if debugmode:
            print(f"starting analys of pop_size :{num}, run: {lvl}, it: {it}")
            start_mi = time.perf_counter()

        mi, smooth_rate, mi_bin = mi_analysis_dynamic_single(stim, spk_id, spk_t, paras_an, paras_model, debugmode, folder, lvl, it)

        if debugmode:
            end_mi = time.perf_counter()
            print(f"an num {num} lvl {lvl}, it {it} finished, took {end_mi-start_mi:4f} secs, or {(end_mi-start_mi)/60.0:4f} mins")

        mi_workr = mi
        mi_bin_workr = mi_bin

        stim_z = st.zscore(stim, axis=None)
        smooth_rate_z = st.zscore(smooth_rate, axis=None)
        corr_workr = np.corrcoef(stim_z, smooth_rate_z)[0][1]

    return num, lvl, it, mi_workr, mi_bin_workr, corr_workr
