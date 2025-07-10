import json
import time
import os
import pathlib as p
import numpy as np
from ModelBuilder import ModelBuilder
from Experimenter import Experimenter
from Analyzer import Analyzer

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

        self.noise_lvls = np.linspace(self.sim_paras["noiselvl_min"], self.sim_paras["noiselvl_max"], self.sim_paras["steps"])

    def run(self):
        
        noiselvls_comp = self.noise_lvls.tolist() # converting the numpy array into a list to be inserted in dict
        print(f"starting simulations at noise levels: {noiselvls_comp}")

        self.sim_settings["noise_levels"] = noiselvls_comp
        self.sim_settings["model_settings"] = self.mod_paras
        self.sim_settings["simulation_analysis_parameters"] = self.an_paras

        run = 0
        for lvl in self.noise_lvls:
            run += 1 # or could make it start from 0 to follow python indexing?

            build_plan = ModelBuilder(self.mod_paras)
            model, noise = build_plan.build(lvl) # also take the noise level in the current model, only to write it in the run setings

            experiment = Experimenter(model, self.exp_paras, self.folder, run, noise, self.spk_rec_steps)
            data_path = experiment.run()

            analysis = Analyzer(data_path, self.an_paras, self.mod_paras)
            analysis.analyze()

        with open(os.path.join(self.folder, 'sim_settings.json'), 'w') as fp:
            json.dump(self.sim_settings, fp, indent=4)
