import json
import time
import os
import numpy as np
from ModelBuilder import ModelBuilder
from Experimenter import Experimenter
from SDFplotter import SDFplotter

class Simulator:

    def __init__(self, parameters:str):

        self.paras_path = parameters
        self.sim_settings = dict()

        with open(self.paras_path) as f:
            parameters = json.load(f)
        
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        simdirname = "simulations"
        current_dir = os.path.join(current_dir, simdirname)
        now = time.strftime("%Y%m%d_%H%M%S")
        dirname = f"sim_{now}"
        self.folder = os.path.join(current_dir, dirname)
        os.makedirs(self.folder)

        self.mod_paras = parameters["model_parameters"]
        self.exp_paras = parameters["experiment_parameters"]
        self.an_paras = parameters["analysis_parameters"]
        self.sim_paras = parameters["simulation_parameters"]
        self.spk_rec_steps = self.mod_paras["spk_rec_steps"] # ugly way to pass model recording steps to the Experimenter
        self.exp_1 = self.exp_paras["experiment_concurrent"]
        self.exp_2 = self.exp_paras["experiment_separate"]

        self.noise_lvls = np.linspace(self.sim_paras["noiselvl_min"], self.sim_paras["noiselvl_max"], self.sim_paras["steps"])

    def run(self):
        
        noiselvls_comp = self.noise_lvls.tolist() # converting the numpy array into a list to be inserted in dict
        print(f"starting simulations at noise levels: {noiselvls_comp}")

        self.sim_settings["noise_levels"] = noiselvls_comp
        self.sim_settings["model_settings"] = self.mod_paras
        self.sim_settings["simulation_analysis_parameters"] = self.an_paras
        
        build_plan = ModelBuilder(self.mod_paras)
        model = build_plan.build()

        run = 0
        for lvl in self.noise_lvls:
            run += 1 # or could make it start from 0 to follow python indexing?

            experiment = Experimenter(model, self.exp_paras, self.folder, run, lvl, self.spk_rec_steps)
            data_path = experiment.run(self.exp_1, self.exp_2)

            sdf = SDFplotter(data_path, self.an_paras, self.mod_paras)
            sdf.plot()

        with open(os.path.join(self.folder, 'sim_settings.json'), 'w') as fp:
            json.dump(self.sim_settings, fp, indent=4)
