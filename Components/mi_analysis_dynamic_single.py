import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.ndimage import gaussian_filter1d

def mi_analysis_dynamic_single(stim, spk_id, spk_t, paras):

    k = int(paras["k_neighbors"])
    stim_raw = stim
    spk_t = spk_t
    spk_id = spk_id

    sigma = paras["sigma_kernel_ms"]
    neuron2analyze = int(paras["neuron_idx_1based"] - 1)
    sim_time = paras["sim_time_secs"]
    dt = paras["timestep_ms"]
    dt_secs = dt / 1000
    n_elements = int(sim_time*1000/dt)
    step_ds = int(paras["downsample_step_ms"]/dt)

    spks = spk_t[spk_id == neuron2analyze]
    spk_array = np.zeros(n_elements)
    idxs = (spks/dt).round().astype(np.int64)
    if len(idxs) > 0:
        spk_array[idxs] = 1
    else: print("Warning: it seems output neuron never fired")

    sigma_bins = sigma/dt

    smooth_rate = gaussian_filter1d(spk_array, sigma=sigma_bins, mode="constant") / dt_secs

    rate_ds = smooth_rate[::step_ds]
    stim_ds = stim_raw[::step_ds]

    mi = mutual_info_regression(stim_ds.reshape(-1,1), rate_ds, n_neighbors = k)[0]

    return mi
