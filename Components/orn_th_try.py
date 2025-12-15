import numpy as np
from matplotlib import pyplot as plt
import pygenn
import os

os.environ['CUDA_PATH'] = '/usr/local/cuda'
pygenn.genn_model.backend_modules["CUDA"] = pygenn.cuda_backend

adapt_lifi = pygenn.create_neuron_model(
                "adaptive_LIF",
                params = [
                    "C_mem", "V_reset", "V_thresh", "V_leak", "g_leak", "r_scale", "g_adapt", "V_adapt", "tau_adapt", "noise_A"
                ],
                vars = [
                    ("V", "scalar"), ("a", "scalar")
                ],
                sim_code = """
                V += (-g_leak*(V-V_leak) - g_adapt*a*(V-V_adapt) + r_scale*Isyn + noise_A*gennrand_normal())*dt/C_mem;
                a += -a*dt/tau_adapt;
                """,
                threshold_condition_code = """
                V >= V_thresh
                """,
                reset_code = """
                V = V_reset;
                a += 0.5;
                """
            )

param_orn = {
            "C_mem": 1.0,
            "V_reset": -70.0,
            "V_thresh": -40.0,
            "V_leak": -60.0,
            "g_leak": 0.01,
            "r_scale": 10.0,
            "g_adapt": 0.0015,
            "V_adapt": -70.0,
            "tau_adapt": 1000.0,
            "noise_A": 0.0,
        }

init_param_orn = {
            "V": -60.0,
            "a": 0.0
        }

sim_t = 50 * 1000
spk_rec_steps = int(1*1000)

model = pygenn.GeNNModel("float", "orn_th", backend = "CUDA")
model.dt = 0.1

model.add_neuron_population("orn", 1, adapt_lifi, param_orn, init_param_orn)
neurons = model.neuron_populations["orn"]
neurons.spike_recording_enabled = True
model.add_current_source("CurrentSource", "DC", neurons, {"amp": 0.022}, {})

model.build()
model.load(spk_rec_steps)
spike_t = []
while model.t < sim_t:

    model.step_time()
    
    if model.timestep%spk_rec_steps == 0:
        print(model.t)
        model.pull_recording_buffers_from_device()
        spike_t.append(neurons.spike_recording_data[0][0])
model.unload()

num_spikes = sum(len(spike_batch) for spike_batch in spike_t)
rate = (num_spikes / sim_t) * 1000
print(f"fr = {rate}Hz")
