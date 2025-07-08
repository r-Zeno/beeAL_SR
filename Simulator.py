import json
import numpy as np
from ModelBuilder import ModelBuilder
from Experimenter import Experimenter
from Analyzer import Analyzer

class Simulator:

    def __init__(self, parameters:str):

        self.paras_path = parameters

        with open(self.paras_path) as f:
            parameters = json.load(f)
        
        self.mod_paras = parameters["model_parameters"]
        self.exp_paras = parameters["experiment_parameters"]
        self.an_paras = parameters["analysis_parameters"]
        self.sim_paras = parameters["simulation_parameters"]

        self.noise_lvls = np.linspace(self.sim_paras["noiselvl_min"], self.sim_paras["noiselvl_max"], self.sim_paras["steps"])

    def run(self):
        
        print(f"starting simulations at noise levels: {self.noise_lvls}")
        for lvl in self.noise_lvls:

            build_plan = ModelBuilder(self.mod_paras)
            model = build_plan.build(lvl)

            experiment = Experimenter(model, self.exp_paras)
            data_path = experiment.run()

            analysis = Analyzer(data_path, self.an_paras, self.mod_paras)
            analysis.analyze()


