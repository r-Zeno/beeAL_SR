import os
import numpy as np
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

@jit(nopython=True)
def vp_metric(train_1, train_2, cost):
    """
    Computes the Victor-Purpura distance, given a pair of spike train timings and a cost (q*Dt), numba optimized.
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
        padding = 200
    else: padding = 0

    # probably all these nested loops should be vectorized, but its the last thing to spend time on
    runs_baseline = {}
    runs_stimulation = {}
    for run in paths:

        run_name = os.path.basename(run)
        curr_run_baseline = {}
        curr_run_stimulation = {}

        for odor in paras["odors"]:

            curr_odor_baseline = {}
            curr_odor_stimulation = {}

            for pop in paras["which_pop"]: # useless for now since only 1 pop, included for scalability

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

                    elif paras["start_stim"] < t < paras["end_stim"]+padding:

                        if key not in curr_odor_stimulation[pop]:
                            curr_odor_stimulation[pop][key] = []

                        curr_odor_stimulation[pop][key].append(t)
                    else: pass

                final_spk_baseline = {}
                final_spk_stimulation = {}
                for i in range(paras["pop_number"]):
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

def exploratory_plots(path, meanvp, singlevp, selected_neurons, rate_delta, flat_rate_base, flat_rate_stim, relative_rate_delta, rate_delta_odorsdiff, relative_rate_delta_odorsdiff, paras_sim, paras_dist):

    x = np.linspace(paras_sim["noiselvl_min"], paras_sim["noiselvl_max"], paras_sim["steps"])
    fig1, ax1 = plt.subplots()
    ax1.scatter(x, meanvp)
    ax1.set_title("mean distance at n levels")
    ax1.set_ylabel("VP-distance (normalized)")
    ax1.set_xlabel("noise level (scaling)")
    plt.savefig(os.path.join(path, "mean_distance.png"))
    plt.close()

    num_n = 800
    neurons = np.zeros(num_n)
    neurons[selected_neurons] = 1
    xn = np.arange(num_n)
    fig2, ax2 = plt.subplots()
    ax2.plot(xn, neurons)
    ax2.set_title("selected neurons")
    ax2.set_yticks([0,1])
    ax2.set_xlabel("neurons")
    plt.savefig(os.path.join(path, "selected_neurons.png"))
    plt.close()

    singlevp = np.array(singlevp) # otherwise can't transpose
    fig3, ax3 = plt.subplots()
    heat1 = ax3.imshow(singlevp.T, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    fig3.colorbar(heat1, ax=ax3)
    ax3.set_title("distance values per neuron")
    ax3.set_ylabel("PN neurons")
    ax3.set_xlabel("noise level (scaling)")
    plt.savefig(os.path.join(path, "distances_single.png"))
    plt.close()

    fig10, ax10= plt.subplots()
    heat8 = ax10.imshow(flat_rate_base["odor_1"], cmap="viridis", aspect="auto")
    fig10.colorbar(heat8, ax=ax10)
    ax10.set_title("baseline firing rate - odor1")
    ax10.set_ylabel("PN neurons")
    ax10.set_xlabel("noise level (scaling)")
    plt.savefig(os.path.join(path, "flat_base_rate_od1.png"))
    plt.close()

    fig11, ax11= plt.subplots()
    heat9 = ax11.imshow(flat_rate_base["odor_2"], cmap="viridis", aspect="auto")
    fig11.colorbar(heat9, ax=ax11)
    ax11.set_title("baseline firing rate - odor2")
    ax11.set_ylabel("PN neurons")
    ax11.set_xlabel("noise level (scaling)")
    plt.savefig(os.path.join(path, "flat_base_rate_od2.png"))
    plt.close()

    fig12, ax12= plt.subplots()
    heat10 = ax12.imshow(flat_rate_stim["odor_1"], cmap="viridis", aspect="auto")
    fig12.colorbar(heat10, ax=ax12)
    ax12.set_title("baseline firing rate - odor2")
    ax12.set_ylabel("PN neurons")
    ax12.set_xlabel("noise level (scaling)")
    plt.savefig(os.path.join(path, "flat_stim_rate_od1.png"))
    plt.close()

    fig13, ax13= plt.subplots()
    heat11 = ax13.imshow(flat_rate_stim["odor_2"], cmap="viridis", aspect="auto")
    fig13.colorbar(heat11, ax=ax13)
    ax13.set_title("baseline firing rate - odor2")
    ax13.set_ylabel("PN neurons")
    ax13.set_xlabel("noise level (scaling)")
    plt.savefig(os.path.join(path, "flat_stim_rate_od2.png"))
    plt.close()

    fig4, ax4= plt.subplots()
    heat2 = ax4.imshow(rate_delta["odor_1"], cmap="viridis", aspect="auto")
    fig4.colorbar(heat2, ax=ax4)
    ax4.set_title("change in firing rate - odor1")
    ax4.set_ylabel("PN neurons")
    ax4.set_xlabel("noise level (scaling)")
    plt.savefig(os.path.join(path, "delta_r_odor1.png"))
    plt.close()

    fig8, ax8= plt.subplots()
    heat6 = ax8.imshow(rate_delta["odor_2"], cmap="viridis", aspect="auto")
    fig8.colorbar(heat6, ax=ax8)
    ax8.set_title("change in firing rate - odor2")
    ax8.set_ylabel("PN neurons")
    ax8.set_xlabel("noise level (scaling)")
    plt.savefig(os.path.join(path, "delta_r_odor2.png"))
    plt.close()

    fig5, ax5 = plt.subplots()
    heat3 = ax5.imshow(relative_rate_delta["odor_1"], cmap="viridis", aspect="auto", vmin=-1.0, vmax=1.0)
    fig5.colorbar(heat3, ax=ax5)
    ax5.set_title("change in firing rate relative to baseline - odor1")
    ax5.set_ylabel("PN neurons")
    ax5.set_xlabel("noise level (scaling)")
    plt.savefig(os.path.join(path, "relative_delta_r_odor1.png"))
    plt.close()

    fig9, ax9 = plt.subplots()
    heat7 = ax9.imshow(relative_rate_delta["odor_2"], cmap="viridis", aspect="auto", vmin=-1.0, vmax=1.0)
    fig9.colorbar(heat7, ax=ax9)
    ax9.set_title("change in firing rate relative to baseline - odor1")
    ax9.set_ylabel("PN neurons")
    ax9.set_xlabel("noise level (scaling)")
    plt.savefig(os.path.join(path, "relative_delta_r_odor2.png"))
    plt.close()
    
    fig6, ax6 = plt.subplots()
    heat4 = ax6.imshow(rate_delta_odorsdiff, cmap="viridis", aspect="auto")
    fig6.colorbar(heat4, ax=ax6)
    ax6.set_title("firing rate difference between odors")
    ax6.set_ylabel("PN neurons")
    ax6.set_xlabel("noise level (scaling)")
    plt.savefig(os.path.join(path, "rate_delta_odorsdiff.png"))
    plt.close()

    fig7, ax7 = plt.subplots()
    heat5 = ax7.imshow(relative_rate_delta_odorsdiff, cmap="viridis", aspect="auto")
    fig7.colorbar(heat5, ax=ax7)
    ax7.set_title("firing rate relative difference between odors")
    ax7.set_ylabel("PN neurons")
    ax7.set_xlabel("noise level (scaling)")
    plt.savefig(os.path.join(path, "relative_rate_delta_odorsdiff.png"))
    plt.close()