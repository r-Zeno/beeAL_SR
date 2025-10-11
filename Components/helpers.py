import os
import numpy as np
import pdv_cuda
from matplotlib import pyplot as plt
from numba import jit

def gauss_odor(n_glo: int, m: float, sd_b: float, a_rate: float, A: float=1.0, clip: float=0.0, m_a: float=0.25, sd_a: float=0.0, min_a: float=0.01, max_a: float=0.05, rand_a: bool=False ,hom_a: bool=True) -> np.array:
    """
    Generates odor binding rate and activation rate, following a gaussian for each glom,
    based on its distance (only for binding rate if hom_a=True) from the maximally responding glom = m (midpoint of gaussian).
    - n_glo: number of glomeruli
    - m: midpoint of gaussian profile for binding rates (also the idx of the maximum responding glom to that odor)
    - sd_b: the standardard deviation of gaussian distr of binding rates (so how specific is the glom response to the odor)
    - a_rate: activation rate for homogenous odor act rate. used only if rand_a=False (it is by default)
    - A: amplitude of gaussian
    - clip: cut-off for binding threshold, if a glom following the assignement based on distance would have a lower binding rate its set to 0
    - m_a: mean activation rate
    - sd_a: standard deviation of activation rate
    - min_a: minimum act rate value
    - max_a: maximum act rate value
    - rand_a: whether the activation rate is randomly chosen or specified by "a_rate"
    - hom_a: whether the activation rate is the same for all glomeruli (for the individual odor) 
    """
    odor = np.zeros((n_glo,2)) # row: glom, col: 0 -> binding rate|1 -> activation rate
    d = np.arange(0,n_glo) # representation of glomeruli
    d = d-m # each glom is represented by its distance to the glom 'm'
    d = np.minimum(np.abs(d), np.abs(d+n_glo)) # pathfinding/clock-like solution to find the distance of each glom to glom[m]
    d = np.minimum(np.abs(d), np.abs(d-n_glo))
    od = np.exp(-np.power(d,2)/(2*np.power(sd_b,2)))
    od *= np.power(10,A)
    od[od > 0] = od[od > 0] + clip
    odor[:,0] = od # assigning binding rates
    if rand_a:
        if hom_a:
            a = 0.0
            while a < min_a or a > max_a:
                a = np.random.normal(m_a,sd_a)
            odor[:,1] = a
        else:
            for i in range(n_glo):
                a = 0.0
                while a < min_a or a > max_a:
                    a = np.random.normal(m_a,sd_a)
                odor[i,1] = a
    else:
        odor[:,1] = a_rate

    return odor

def set_odor_simple(ors, slot, odor, con, hill):
    """
    setting parameters of ors for the chosen odor (effectively "presenting"the odor to the ors).
    Difference from the og function: it autonomously views and pushes the variables to the ors.
    - ors: the population of ors (receptors)
    - odor: 
    - slot: the odor slot to use for each or (each has 3, but can be easily augmented)
    - con: concentration of odor: 1e-7 to 1e-1
    - hill: the hill coefficient
    """
    od = np.squeeze(odor)
    kp1cn = np.power(od[:,0]*con,hill)
    km1 = 0.025
    kp2 = od[:,1]
    km2 = 0.025

    vars_names = [
        f"kp1cn_{slot}",
        f"km1_{slot}",
        f"kp2_{slot}",
        f"km2_{slot}"
    ]

    odor_vars = {
        f"kp1cn_{slot}": kp1cn,
        f"km1_{slot}": km1,
        f"kp2_{slot}": kp2,
        f"km2_{slot}": km2
    }

    for odvar_name, odvalues in odor_vars.items():
        ors.vars[odvar_name].view[:] = odvalues

    for name in vars_names:
        ors.vars[name].push_to_device()

def make_sdf(sT, sID, allID, t0, tmax, dt, sigma):
    """"
    Computes Spike Density Function from spiking data. time x neuron id
    """
    tleft= t0-3*sigma
    tright= tmax+3*sigma
    n= int((tright-tleft)/dt)
    sdfs= np.zeros((n,len(allID)))
    kwdt= 3*sigma
    i= 0
    x= np.arange(-kwdt,kwdt,dt)
    x= np.exp(-np.power(x,2)/(2*sigma*sigma))
    x= x/(sigma*np.sqrt(2.0*np.pi))*1000.0
    if sT is not None:
        for t, sid in zip(sT, sID):
            if (t > t0 and t < tmax): 
                left= int((t-tleft-kwdt)/dt)
                right= int((t-tleft+kwdt)/dt)
                if right <= n:
                    sdfs[left:right,sid]+=x
           
    return sdfs

def force_aspect(ax,aspect):
    """
    Controls aspect ratio of figs
    """
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def glo_avg(sdf: np.ndarray, n):
    """
    Returns the average sdf for glomerolus
    sdf: spike density function (time x neurons)
    n: number of neurons (of the type plotted) per glomerolus
    """
    nglo= sdf.shape[1]//n
    gsdf= np.zeros((sdf.shape[0],nglo))
    for i in range(nglo):
        gsdf[:,i]= np.mean(sdf[:,n*i:n*(i+1)],axis=1)
    return gsdf

@jit(nopython=True) # next step is writing a CUDA kernel for this, passing spike data to c++ code,
# as it takes too much (at least as long as the sim itself) when analyzing all the neurons from all pops
def vp_metric(train_1, train_2, cost):
    """
    Computes the Victor-Purpura distance normalized by spike count, given a pair of spike train timings and a cost (q*Dt), numba optimized.
    - train_1, train_2: timings of spikes
    - cost: the q value
    """
    s_train_i = train_1
    s_train_j = train_2
    q = cost

    nspt_i = len(s_train_i)
    nspt_j = len(s_train_j)

    scr = np.zeros((nspt_i + 1, nspt_j +1))
    scr[:,0] = np.arange(0, nspt_i + 1)
    scr[0,:] = np.arange(0, nspt_j + 1)

    if nspt_i > 0 and nspt_j >0:
        
        for i in np.arange(1, nspt_i + 1):

            for j in np.arange(1, nspt_j + 1):

                scr[i,j] = min(scr[i-1,j]+1, scr[i,j-1]+1, scr[i-1, j-1]+ q * abs(s_train_i[i-1] - s_train_j[j-1]))

    raw_dist = scr[nspt_i, nspt_j]
    norm_dist = raw_dist / (nspt_i + nspt_j)

    return norm_dist

def data_prep4numba_distance(train_1, train_2, cost):
    """
    Assigns a distance value of 0 to empty spike lists and converts existing spike lists to numpy array to help 
    speed up the distance algorithm (numba doesn't know what to do with empty lists).

    Automatically calls the distance computation function.
    """

    if len(train_1) == 0 and len(train_2) == 0:
        return 0.0
    else: return vp_metric(np.array(train_1), np.array(train_2), cost)

def neuron_spikes_assemble(paths, paras, pad:bool):

    if pad:
        padding = 200 # padding lenght selection should be in json but its never used (hardcoded false in simulator)
    else: padding = 0

    # to vectorize if possible
    runs_baseline = {}
    runs_stimulation = {}
    for run in paths:

        run_name = os.path.basename(run)
        curr_run_baseline = {}
        curr_run_stimulation = {}

        for odor in paras["odors"]:

            curr_odor_baseline = {}
            curr_odor_stimulation = {}

            for pop in paras["which_pop"]:

                curr_spike_ts = np.load(os.path.join(run, odor, f"{pop}_spike_t.npy"))
                curr_spike_ids = np.load(os.path.join(run, odor, f"{pop}_spike_id.npy"))
                curr_pop_idxt = np.stack((curr_spike_ids, curr_spike_ts), 1)
                curr_odor_baseline[pop] = {}
                curr_odor_stimulation[pop] = {}

                for (id, t) in curr_pop_idxt:
                    key = int(id)
                    if t < paras["start_stim"]:

                        if key not in curr_odor_baseline[pop]:
                            curr_odor_baseline[pop][key] = []

                        curr_odor_baseline[pop][key].append(t)

                    elif paras["start_stim"]-padding < t < paras["end_stim"]+padding:

                        if key not in curr_odor_stimulation[pop]:
                            curr_odor_stimulation[pop][key] = []

                        curr_odor_stimulation[pop][key].append(t)
                    else: pass

                final_spk_baseline = {}
                final_spk_stimulation = {}
                for i in range(paras["which_pop"][pop][1]):
                    final_spk_baseline[i] = curr_odor_baseline[pop].get(i, [])
                    final_spk_stimulation[i] = curr_odor_stimulation[pop].get(i, [])

                curr_odor_baseline[pop] = final_spk_baseline
                curr_odor_stimulation[pop] = final_spk_stimulation

            curr_run_baseline[odor] = curr_odor_baseline
            curr_run_stimulation[odor] = curr_odor_stimulation

        runs_baseline[run_name] = curr_run_baseline
        runs_stimulation[run_name] = curr_run_stimulation

    spks_split = {}
    spks_split["baseline"] = runs_baseline
    spks_split["stimulation"] = runs_stimulation

    return spks_split

def fire_rate(data:dict, paras):

    baseline_t = paras["start_stim"] / 1000.0 # from ms to s (rate in Hz)
    stimulation_t = (paras["end_stim"] - paras["start_stim"]) / 1000.0
    print(f"DEBUG: Baseline Duration = {baseline_t}s")
    print(f"DEBUG: Stimulation Duration = {stimulation_t}s")
    rates = {}
    for state, runs in data.items():
        rates[state] = {}

        if state == "baseline":
            duration = baseline_t
        else: duration = stimulation_t

        for run_n, odors in runs.items():
            rates[state][run_n] = {}

            for odor_n, pops in odors.items():
                rates[state][run_n][odor_n] = {}

                for pop_n, neurons in pops.items():
                    rates[state][run_n][odor_n][pop_n] = {}

                    for neuron, spikes in neurons.items():

                        curr_rate = len(spikes) / duration
                        rates[state][run_n][odor_n][pop_n][neuron] = curr_rate

    return rates

def toga(data:dict, folder, n_runs):
    """
    to be used everytime data is to be passed to C++/CUDA code for fast computing.
    For now it only serves the use of passing rates.
    Will need to specify grouping behavior, as it may be convenient to create a 2d array for each pop
    or to store all pop together.
    """
    # would be nice to add an intelligent unrolling of the dict, for now it works assuming it follows
    # the structure that is used for all dicts here

    # hardcoding everything, just to start
    # should handle multiple pops, creating a matrix for each!
    pop = "pn" # check the real name of the dict key
    states = ["baseline", "stimulation"]
    n_neurons = 800

    runs_naming = []
    for runs in range(n_runs):
        runs_naming.append(f"run_{runs}")

    odors = ["odor_1", "odor_2"]

    data_extr_od1 = np.zeros((n_neurons, n_runs))
    data_extr_od2 = np.zeros((n_neurons, n_runs))

    for col_idx, run in enumerate(runs_naming):
        for odor in odors:
            for neuron in range(n_neurons): # may be better to join them as 800 x 2000

                rate_b = data["baseline"][run][odor][pop][neuron]
                rate_s = data["stimulation"][run][odor][pop][neuron]

                if odor == "odor_1":
                    data_extr_od1[neuron, col_idx] = rate_s
                elif odor == "odor_2":
                    data_extr_od2[neuron, col_idx] = rate_s

    path1 = os.path.join(folder, "rates_od1.npy")
    path2 = os.path.join(folder, "rates_od2.npy")

    # transposing to runs x neurons: this is better for the gpu, enabling it to make a single memory call to fetch data for every T in a warp
    gpu_data_od1 = data_extr_od1.T
    gpu_data_od2 = data_extr_od2.T
    np.save(path1, gpu_data_od1)
    np.save(path2, gpu_data_od2)

    pdv = pdv_cuda.compute_D_gpu(gpu_data_od1, gpu_data_od2, 1)

    np.save(os.path.join(folder, "pdv.npy"), pdv)

    return None

def exploratory_plots(
        path, pop, pop_nums, meanvp, singlevp, selected_neurons, flat_rate_base_od1, flat_rate_base_od2, 
        flat_rate_stim_od1, flat_rate_stim_od2, rate_delta_od1, rate_delta_od2, relative_rate_delta_od1, 
        relative_rate_delta_od2, rate_delta_odorsdiff, relative_rate_delta_odorsdiff, 
        paras_sim:dict, paras_plots:dict
        ):

    plotnames = []

    x = np.linspace(paras_sim["noiselvl_min"], paras_sim["noiselvl_max"], paras_sim["steps"])
    fig1, ax1 = plt.subplots()
    ax1.scatter(x, meanvp)
    ax1.set_title(f"mean distance at n levels ({pop})")
    ax1.set_ylabel("VP-distance (normalized)")
    ax1.set_xlabel("noise level (scaling)")
    mname = f"mean_distance_{pop}.png"
    plt.savefig(os.path.join(path, mname))
    plt.close()
    plotnames.append(mname)

    num_n = pop_nums[pop][1]
    neurons = np.zeros(num_n)
    neurons[selected_neurons] = 1
    xn = np.arange(num_n)
    fig2, ax2 = plt.subplots()
    ax2.plot(xn, neurons)
    ax2.set_title(f"selected neurons_{pop}")
    ax2.set_yticks([0,1])
    ax2.set_xlabel("neurons")
    selname = f"selected_neurons_{pop}.png"
    plt.savefig(os.path.join(path, selname))
    plt.close()
    plotnames.append(selname)

    for plot in paras_plots:

        p_fname = eval(paras_plots[plot]["filename"])
        numticks = paras_plots[plot]["nticks"]
        ncols = paras_sim["steps"]
        tickpos = np.linspace(0, ncols-1, numticks, dtype=int)
        labels = np.linspace(paras_sim["noiselvl_min"], paras_sim["noiselvl_max"], numticks)
        for i in range(len(labels)):
            labels[i] = round(labels[i], 2)

        fig, ax = plt.subplots()
        heat = ax.imshow(eval(paras_plots[plot]["data"]), cmap=paras_plots[plot]["color_map"], aspect="auto")
        fig.colorbar(heat, ax=ax)
        ax.set_title(eval(paras_plots[plot]["title"]))
        ax.set_ylabel(paras_plots[plot]["ylabel"])
        ax.set_xlabel(paras_plots[plot]["xlabel"])
        ax.set_xticks(tickpos)
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", labelrotation=45)
        plt.savefig(os.path.join(path, p_fname),dpi=paras_plots[plot]["dpi"])
        plt.close()
        plotnames.append(p_fname)

    return plotnames
