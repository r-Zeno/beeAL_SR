import os
import json
import time
import numpy as np
from ModelBuilder import ModelBuilder
from Experimenter import Experimenter
from ExperimentStatic import *
from Analyzer import Analyzer
from SDFplotter import SDFplotter
from DistanceAnalyzer import DistanceAnalyzer
from NeuronSelector import NeuronSelector
from RateAnalyzer import RateAnalyzer
from helpers import exploratory_plots, neuron_spikes_assemble, fire_rate, toga

class Simulator:

    def __init__(self, parameters_path:str):

        self.sim_settings = {}

        with open(parameters_path) as f:
            self.parameters = json.load(f)
        
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        simdirname = "simulations"
        current_dir = os.path.join(current_dir, simdirname)
        now = time.strftime("%Y%m%d_%H%M%S")
        self.dirname = f"sim_{now}"
        self.folder = os.path.join(current_dir, self.dirname)
        os.makedirs(self.folder)

        #self.exp_paras = parameters["experiment_parameters"]
        #self.sdf_paras = parameters["analysis_parameters"]["sdf_parameters"]
        #self.dist_paras = parameters["analysis_parameters"]["distance_parameters"]
        #self.sim_paras = parameters["simulation_parameters"]
        #self.plot_paras = parameters["exp_plot_parameters"]
        #self.spk_rec_steps = self.mod_paras["spk_rec_steps"] # ugly way to pass model recording steps to the Experimenter
        #self.exp_1 = self.exp_paras["experiment_concurrent"]
        #self.exp_2 = self.exp_paras["experiment_separate"]
        #self.debugmode = self.sim_paras["debugmode"]
        #self.selection_criterion = self.sim_paras["selection_criterion"]
        #self.pops = self.dist_paras["which_pop"]

        self.mod_paras = self.parameters["model_parameters"]
        self.sim_paras = self.parameters["simulation_parameters"]
        self.exp_name = self.sim_paras["which_exp"]
        self.debugmode = self.sim_paras["debugmode"]
        self.exp_paras = self.parameters["experiment_parameters"][self.exp_name]
        self.an_paras = self.parameters["analysis_parameters"][self.exp_name]
        self.target_pop = self.exp_paras["stimulus"]["target_pop"]
        self.sim_time_s = self.exp_paras["exp_duration_s"]

    def run(self):
        
        #noiselvls_comp = self.noise_lvls.tolist() # converting the numpy array into a list to be inserted in dict
        #print(f"starting simulations at noise levels: {noiselvls_comp}")

        #self.sim_settings["noise_levels"] = noiselvls_comp
        self.sim_settings["model_settings"] = self.mod_paras
        self.sim_settings["eperiment"] = self.exp_name
        self.sim_settings["experiment_settings"] = self.exp_paras
        single_num = self.exp_paras["orn_n"]["single_num"]
        if single_num:
            pop_numbers = [self.exp_paras["orn_n"]["num_if_single"]]
        else:
            max_range = self.exp_paras["orn_n"]["range"]
            base = self.exp_paras["orn_n"]["base"]
            pwrs = np.arange(1,max_range, dtype=int)
            pop_numbers = base**pwrs

        data_log_full = []

        for n in pop_numbers:

            self.mod_paras["num"]["orn"] = int(n)

            build_plan = ModelBuilder(self.mod_paras, self.exp_name, self.sim_time_s, self.target_pop, self.debugmode)
            model = build_plan.build()

            #data_paths = []
            #run = 0
            #for lvl in self.noise_lvls:
            #    experiment = ExperimentStatic(model, self.exp_paras, self.folder, run, lvl, self.spk_rec_steps, self.debugmode)
            #    data_path = experiment.run(self.exp_1, self.exp_2)
            #    data_paths.append(data_path)
            #    run += 1
            start = time.time()

            experiment = Experimenter(model, self.exp_paras, self.folder, self.exp_name, n, self.debugmode)
            stim_path, curr_data_log = experiment.run()

            data_log_full.extend(curr_data_log)

            end = time.time()
            timetaken_sim = round(end - start, 2)
        print(f"Simulations ended, it took {timetaken_sim:2f} secs or {timetaken_sim/60:2f} mins")

        analysis = Analyzer(self.folder, self.an_paras, self.mod_paras, stim_path, data_log_full, self.exp_name, pop_numbers, self.debugmode)
        res_path = analysis.run()

        if self.debugmode: print(self.dirname)

        with open(os.path.join(self.folder, "sim_settings.json"), "w") as fp:
            json.dump(self.parameters, fp, indent=4)

        ####### All this must be handled by Experimenter, Analyzer and Plotter #######

        #spk_split = neuron_spikes_assemble(data_paths, self.dist_paras, pad=False)

        #rates = fire_rate(spk_split, self.dist_paras)

        #rate_init = RateAnalyzer(rates, self.dist_paras)
        #flat_rate_base, flat_rate_stim, rate_delta, relative_rate_delta, rate_delta_odorsdiff, relative_rate_delta_odorsdiff = rate_init.get_rate_diffs()

        #selector_init = NeuronSelector(self.dist_paras, rates, rate_delta_odorsdiff, self.noise_lvls, self.selection_criterion)
        #neurons2analyze = selector_init.select()

        #print("Starting analysis...")
        #start = time.time()
        #if self.sim_paras["sdf"]:
        #    for path in data_paths:
        #        sdf = SDFplotter(path, self.sdf_paras, self.mod_paras)
        #        sdf.plot()

        #if self.sim_paras["dist"]:
        #    single_vpdist = {} # for debugging
        #    means_vpdist = {}

        #    for path in data_paths:
        #        runname = os.path.basename(path)
        #        single_vpdist[runname] = {}
        #        means_vpdist[runname] = {}

        #        for pop in self.pops:

        #            vpdist_init = DistanceAnalyzer(path, self.dist_paras, pop, neurons2analyze[pop])
        #            dist_result, dist_single = vpdist_init.compute_distance()
        #            single_vpdist[runname][pop] = dist_single
        #            means_vpdist[runname][pop] = dist_result
        #end = time.time()
        #timetaken_an = round(end - start,2)
        #print(f"VP dist Analysis ended,\n Time spent in sim: {timetaken_sim}s | {round(timetaken_sim/60,2)} min, time spent computing distances: {timetaken_an}s | {round(timetaken_an/60,2)} min")

        #if self.sim_paras["dist"]:
        #    plot_names = {}

        #    for pop in self.pops:

        #        curr_vpmean_runxpop = [means_vpdist[run][pop] for run in means_vpdist] 
        #        curr_vpsingle_runxpop = [single_vpdist[run][pop] for run in single_vpdist]
        #        curr_vpsingle_prep = np.array(curr_vpsingle_runxpop).T
        #        curr_flatrate_base_od1 = flat_rate_base["odor_1"][pop]
        #        curr_flatrate_base_od2 = flat_rate_base["odor_2"][pop]
        #        curr_flatrate_stim_od1 = flat_rate_stim["odor_1"][pop]
        #        curr_flatrate_stim_od2 = flat_rate_stim["odor_2"][pop]
        #        curr_rate_delta_od1 = rate_delta["odor_1"][pop]
        #        curr_rate_delta_od2 = rate_delta["odor_2"][pop]
        #        curr_relrate_delta_od1 = relative_rate_delta["odor_1"][pop]
        #        curr_relrate_delta_od2 = relative_rate_delta["odor_2"][pop]
        #        curr_rate_delta_odorsdiff = rate_delta_odorsdiff[pop]
        #        curr_relrate_delta_odorsdiff = relative_rate_delta_odorsdiff[pop]

        #        p_names = exploratory_plots(
        #            self.folder, pop, self.pops, curr_vpmean_runxpop, curr_vpsingle_prep, neurons2analyze[pop], 
        #            curr_flatrate_base_od1, curr_flatrate_base_od2, curr_flatrate_stim_od1, curr_flatrate_stim_od2, 
        #            curr_rate_delta_od1, curr_rate_delta_od2, curr_relrate_delta_od1, curr_relrate_delta_od2, 
        #            curr_rate_delta_odorsdiff, curr_relrate_delta_odorsdiff, self.sim_paras, self.plot_paras
        #            )

        #        plot_names[pop] = p_names

        #        np.save(os.path.join(self.folder, f"mean_vp_dist_x_noiselvls_{pop}.npy"), curr_vpmean_runxpop)
        #        np.save(os.path.join(self.folder, f"single_vp_dist_values_{pop}.npy"), curr_vpsingle_runxpop)
            
        #    toga(rates, self.folder, self.sim_paras["steps"])

        #with open(os.path.join(self.folder, "plot_names.json"), "w") as fp:
        #    json.dump(plot_names, fp, indent=4)
        #with open(os.path.join(self.folder, "sim_settings.json"), "w") as fp:
        #    json.dump(self.parameters, fp, indent=4)
