import numpy as np
from matplotlib import pyplot as plt
from pygenn.genn_model import GeNNModel
import time
import os
import json
from helpers import gauss_odor, set_odor_simple, make_sdf
from ModelBuilder import *

debug = True

with open("parameters_static.json") as f:
    paras = json.load(f)

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
simdirname = "simulations"
current_dir = os.path.join(current_dir, simdirname)
now = time.strftime("%Y%m%d_%H%M%S")
dirname = f"sim_{now}"
folder = os.path.join(current_dir, dirname)
os.makedirs(folder)

mod_paras = paras["model_parameters"]

build_plan = ModelBuilder(
    mod_paras,
    "static",
    1000,
    "orn",
    True
)
model = build_plan.build()

############## creating odors
odors= []
odor_stable_params = paras["odors_stable_params"]

od1_m = 119
od1_sd_b = 1
od1 = gauss_odor(n_glo=paras["num_glo"], m = od1_m, sd_b= od1_sd_b, a_rate= 0.1, A= 1)
odors.append(np.copy(od1))

od2_m = 121
od2_sd_b = 1
od2 = gauss_odor(n_glo=paras["num_glo"], m = od2_m, sd_b= od2_sd_b, a_rate= 0.1, A= 1)
odors.append(np.copy(od2))

# defining Hill coefficient
hill_exp= np.random.uniform(0.95, 1.05, paras["num_glo"])
np.save(os.path.join(folder,"_hill"), hill_exp)

noise_lvls = [0.0, 1.0]
noisy_pops = ["orn", "pn", "ln"]

pop_to_rec = ["pn"]

spike_t = []
spike_id = []
if debug: print(spike_t)
if debug: print(spike_id)

base = np.power(10,0.25)
c = 12
on = 1e-7*np.power(base,c)
off = 0.0
odor_slot = 0 # more than one odor slot in ors so must specify

baseline = 1000
stim_duration = 3000
t_odor_on = baseline
sim_time = t_odor_on + stim_duration

ors_pop = model.neuron_populations["or"]

start = time.time()
for odor in odors:
    i += 1

    int_t = 0
    odor_applied = False
    odor_removed = False

    model.load(num_recording_timesteps=1000)
    
    corr_noise_lvl = noise_lvls[0]/np.sqrt(model.dt)
    print(f"applying noise lvl {noise_lvls[0]} (corrected: {corr_noise_lvl})")

    for pop in noisy_pops:
        popobj = model.neuron_populations.get(pop)
        if popobj is not None:
            popobj.set_dynamic_param_value("noise_A", corr_noise_lvl)
            if debug: print(f"injecting noise into {popobj}")

    f_name = f"odor_{str(i)}"
    res_folder = os.path.join(folder, f_name)

    while model.t < sim_time:

        if not odor_applied and model.t >= t_odor_on:
            if debug: print(f"applying odor at time {model.t}")
            set_odor_simple(ors_pop, odor_slot, odor, on, hill_exp)
            odor_applied = True

        model.step_time()
        int_t =+ 1

        if int_t%1000 == 0:
            model.pull_recording_buffers_from_device()

            pop_to_pull = model.neuron_populations["pn"]
            if (pop_to_pull.spike_recording_data[0][0].size > 0):
                spike_t.append(pop_to_pull.spike_recording_data[0][0])
                spike_id.append(pop_to_pull.spike_recording_data[0][0])
                if debug: print(f"spiked fetched at time {model.t}")
            else:
                if debug: print(f"no spikes at time {model.t}")

    model.unload()
    os.makedirs(res_folder)
    np.save(os.path.join(res_folder, "clean_pn_spike_t.npy"), spike_t)
    np.save(os.path.join(res_folder, "clean_pn_spike_id.npy"), spike_id)

    sigma = 20
    dt = 1
    tleft = 0 - 3 * sigma
    tright = sim_time + 3 * sigma
    n = int((tright - tleft) / dt)
    time_vector = np.arange(n) * dt + tleft
    allID = list(range(mod_paras["num"]["glom"]*mod_paras["num"]["pn"]))

    sdfs = make_sdf(spike_t, spike_id, allID, 0, sim_time, dt, sigma)
    np.save(os.path.join(res_folder, "sdf_clean.npy"),sdfs)

for odor in odors:
    i += 1

    int_t = 0
    odor_applied = False
    odor_removed = False

    model.load(num_recording_timesteps=1000)
    
    corr_noise_lvl = noise_lvls[0]/np.sqrt(model.dt)
    print(f"applying noise lvl {noise_lvls[0]} (corrected: {corr_noise_lvl})")

    for pop in noisy_pops:
        popobj = model.neuron_populations.get(pop)
        if popobj is not None:
            popobj.set_dynamic_param_value("noise_A", corr_noise_lvl)
            if debug: print(f"injecting noise into {popobj}")

    f_name = f"odor_{str(i)}"
    res_folder = os.path.join(folder, f_name)

    while model.t < sim_time:

        if not odor_applied and model.t >= t_odor_on:
            if debug: print(f"applying odor at time {model.t}")
            set_odor_simple(ors_pop, odor_slot, odor, on, hill_exp)
            odor_applied = True

        model.step_time()
        int_t =+ 1

        if int_t%1000 == 0:
            model.pull_recording_buffers_from_device()

            pop_to_pull = model.neuron_populations["pn"]
            if (pop_to_pull.spike_recording_data[0][0].size > 0):
                spike_t.append(pop_to_pull.spike_recording_data[0][0])
                spike_id.append(pop_to_pull.spike_recording_data[0][0])
                if debug: print(f"spiked fetched at time {model.t}")
            else:
                if debug: print(f"no spikes at time {model.t}")

    model.unload()
    os.makedirs(res_folder)
    np.save(os.path.join(res_folder, "noisy_pn_spike_t.npy"), spike_t)
    np.save(os.path.join(res_folder, "noisy_pn_spike_id.npy"), spike_id)

    sigma = 20
    dt = 1
    tleft = 0 - 3 * sigma
    tright = sim_time + 3 * sigma
    n = int((tright - tleft) / dt)
    time_vector = np.arange(n) * dt + tleft
    allID = list(range(mod_paras["num"]["glom"]*mod_paras["num"]["pn"]))

    sdfs = make_sdf(spike_t, spike_id, allID, 0, sim_time, dt, sigma)
    np.save(os.path.join(res_folder, "sdf_noisy.npy"),sdfs)

plt.figure(figsize=(10, 5))
# Transpose sdfs with .T so neurons are on the Y-axis and time is on the X-axis
plt.imshow(sdfs.T, aspect='auto', cmap='viridis', 
        extent=[tleft, tright, len(allID)-0.5, -0.5], origin='upper')
plt.colorbar(label='Firing Rate')
plt.xlim(t_odor_on, sim_time)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron ID')
plt.title('Population Spike Density')
plt.show()
