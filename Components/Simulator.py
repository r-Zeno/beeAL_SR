import os
import json
import time
import numpy as np
from ModelBuilder import ModelBuilder
from Experimenter import Experimenter
from SDFplotter import SDFplotter
from DistanceAnalyzer import DistanceAnalyzer
from NeuronSelector import NeuronSelector
from RateAnalyzer import RateAnalyzer
from helpers import exploratory_plots, neuron_spikes_assemble, fire_rate

class Simulator:

    def __init__(self, parameters_path:str):

        self.paras_path = parameters_path
        self.sim_settings = dict()

        with open(self.paras_path) as f:
            self.parameters = json.load(f)
        
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        simdirname = "simulations"
        current_dir = os.path.join(current_dir, simdirname)
        now = time.strftime("%Y%m%d_%H%M%S")
        dirname = f"sim_{now}"
        self.folder = os.path.join(current_dir, dirname)
        os.makedirs(self.folder)

        self.mod_paras = self.parameters["model_parameters"]
        self.exp_paras = self.parameters["experiment_parameters"]
        self.sdf_paras = self.parameters["analysis_parameters"]["sdf_parameters"]
        self.dist_paras = self.parameters["analysis_parameters"]["distance_parameters"]
        self.sim_paras = self.parameters["simulation_parameters"]
        self.plot_paras = self.parameters["exp_plot_parameters"]
        self.spk_rec_steps = self.mod_paras["spk_rec_steps"] # ugly way to pass model recording steps to the Experimenter
        self.exp_1 = self.exp_paras["experiment_concurrent"]
        self.exp_2 = self.exp_paras["experiment_separate"]
        self.debugmode = self.sim_paras["debugmode"]
        self.selection_criterion = self.sim_paras["selection_criterion"]

        self.noise_lvls = np.linspace(self.sim_paras["noiselvl_min"], self.sim_paras["noiselvl_max"], self.sim_paras["steps"])

    def run(self):
        
        noiselvls_comp = self.noise_lvls.tolist() # converting the numpy array into a list to be inserted in dict
        start = time.time()
        print(f"starting simulations at noise levels: {noiselvls_comp}")

        self.sim_settings["noise_levels"] = noiselvls_comp
        self.sim_settings["model_settings"] = self.mod_paras
        self.sim_settings["simulation_analysis_parameters"] = self.sdf_paras
        
        build_plan = ModelBuilder(self.mod_paras)
        model = build_plan.build()

        if self.sim_paras["dist"]:
            single_vpdist = [] # for debugging
            means_vpdist = []

        data_paths = []
        run = 0
        for lvl in self.noise_lvls:
            experiment = Experimenter(model, self.exp_paras, self.folder, run, lvl, self.spk_rec_steps, self.debugmode)
            data_path = experiment.run(self.exp_1, self.exp_2)
            data_paths.append(data_path)
            run += 1

        end = time.time()
        timetaken_sim = round(end - start,2)
        print(f"Simulations ended, it took {timetaken_sim}")

        spk_split = neuron_spikes_assemble(data_paths, self.dist_paras, pad=False)
        rates = fire_rate(spk_split, self.dist_paras)

        rate_init = RateAnalyzer(rates, self.dist_paras)
        flat_rate_base, flat_rate_stim, rate_delta, relative_rate_delta, rate_delta_odorsdiff, relative_rate_delta_odorsdiff = rate_init.get_rate_diffs()

        selector_init = NeuronSelector(self.dist_paras, rates, rate_delta_odorsdiff, self.noise_lvls, self.selection_criterion)
        neurons2analyze = selector_init.select()

        print("Starting analysis...")
        start = time.time()
        if self.sim_paras["sdf"]:
            for path in data_paths:
                sdf = SDFplotter(path, self.sdf_paras, self.mod_paras)
                sdf.plot()
        
        if self.sim_paras["dist"]:
            for path in data_paths:
                vpdist_init = DistanceAnalyzer(path, self.dist_paras, neurons2analyze)
                dist_result, dist_single = vpdist_init.compute_distance()
                single_vpdist.append(dist_single)
                means_vpdist.append(dist_result)
        end = time.time()
        timetaken_an = round(end - start,2)
        print(f"Analysis ended,\n Time spent in sim: {timetaken_sim}s | {round(timetaken_sim/60,2)} min, time spent computing distances: {timetaken_an}s")

        if self.sim_paras["dist"]:
            exploratory_plots(self.folder, means_vpdist, single_vpdist, neurons2analyze, rate_delta, 
                              flat_rate_base, flat_rate_stim, relative_rate_delta, rate_delta_odorsdiff, 
                              relative_rate_delta_odorsdiff, self.sim_paras, self.plot_paras)
        
        if self.sim_paras["dist"]:
            np.save(os.path.join(self.folder, "mean_vp_dist_x_noiselvls.npy"), means_vpdist)
            np.save(os.path.join(self.folder, "single_vp_dist_values.npy"), single_vpdist)

        with open(os.path.join(self.folder, 'sim_settings.json'), 'w') as fp:
            json.dump(self.parameters, fp, indent=4)
