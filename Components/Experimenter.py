import numpy as np
import os
import time
from pygenn.genn_model import GeNNModel
from ExperimentStatic import *
from ExperimentDynamicSingle import *

class Experimenter:

    def __init__(self, model:GeNNModel, exp_paras:dict, folder:str, spk_rec_steps, debugmode:bool):

        self.data_paths = []
        self.folder = folder
        self.model = model
        self.spk_rec_steps = spk_rec_steps
        self.debug = debugmode

        exp_type = exp_paras["which_exp"]
        self.paras = exp_paras[exp_type]
        # assuming all possible exp define runs and trial numbers
        self.runs = self.paras["runs"]
        self.trials = self.paras["trials"]
        

        

    def run(self):

        match self.which_exp:

            case "static_single":
                exp = ExperimentStatic()
                exp.run()
            case "dynamic_single":
            
                exp = 0
                for i in range(self.runs):

                    for j in range(self.trials):

                        exp = ExperimentDynamicSingle(self.paras, self.model)
                        stim, spk_id, spk_t = exp.run(i,j)

                        


                
            case _: raise ValueError("invalid experiment selected, check json")

