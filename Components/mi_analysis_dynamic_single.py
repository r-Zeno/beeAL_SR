import numpy as np
import math
import os
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.ndimage import gaussian_filter1d

def mi_analysis_dynamic_single(stim, spk_id, spk_t, paras_an, paras_model, debug, folder, lvl, it):

    k = int(paras_an["k_neighbors"])
    stim_raw = stim
    spk_t = spk_t
    spk_id = spk_id

    if paras_an["neuron_to_analyze"]["auto"]:
        n_glom = paras_model["num"]["glom"]
        n_pn = paras_model["num"]["pn"]

        target_glom = int(math.ceil(n_glom/2) -1)
        neuron2analyze = int(max(0, target_glom*n_pn))
    else: neuron2analyze = int(paras_an["neuron_to_analyze"]["custom_idx_1based"] -1)

    sigma = paras_an["sigma_kernel_ms"]
    sim_time = paras_an["sim_time_secs"]
    dt = paras_an["timestep_ms"]
    dt_secs = dt / 1000
    n_elements = int(sim_time*1000/dt)
    step_ds = int(paras_an["downsample_step_ms"]/dt)

    spks = spk_t[spk_id == neuron2analyze]
    spk_array = np.zeros(n_elements)
    idxs = (spks/dt).round().astype(np.int64)
    if len(idxs) > 0:
        spk_array[idxs] = 1
    else: print("Warning: it seems output neuron never fired")

    sigma_bins = sigma/dt

    smooth_rate = (gaussian_filter1d(spk_array, sigma=sigma_bins, mode="constant") / dt_secs).astype(np.float32)

    if debug: np.save(os.path.join(folder, f"smoothed_rate_lvl{lvl}_it{it}.npy"), smooth_rate)

    rate_ds = smooth_rate[::step_ds].copy()
    stim_ds = stim_raw[::step_ds].copy()

    mi = mutual_info_regression(stim_ds.reshape(-1,1), rate_ds, n_neighbors = k)[0]
    
    bin = int(paras_an["bin_size_ms"]/dt)
    stim_ds_dis = np.zeros(shape=int(n_elements/bin))
    spks_ds_dis = np.zeros(shape=int(n_elements/bin))

    for i in range(0, n_elements, bin):
        values = stim_raw[i:i+bin]
        stim_ds_dis[i//bin] = np.mean(values)

        spks = np.sum(spk_array[i:i+bin])
        spks_ds_dis[i//bin] = int(spks)

    mi_dis = mutual_info_classif(stim_ds_dis.reshape(-1,1), spks_ds_dis, discrete_features=False,n_neighbors=k)[0]

    return mi, smooth_rate, mi_dis
