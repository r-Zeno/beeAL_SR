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

        self.exp_type = exp_paras["which_exp"]
        self.paras = exp_paras[self.exp_type]
        # assuming all possible exp define runs and trial numbers
        self.runs = self.paras["runs"]
        self.trials = self.paras["trials"]

        self.path = os.path.join(folder, self.exp_type)

    def run(self):

        match self.exp_type:

            case "static_single":
                exp = ExperimentStatic()
                exp.run()

            case "dynamic_single":
                
                stim_gen = None
                exp = 0
                for i in range(self.runs):

                    dirname = os.path.join(self.folder,f"lvl_{i}")

                    for j in range(self.trials):

                        exp = ExperimentDynamicSingle(self.paras, self.model, stim_gen)
                        stim, spk_id, spk_t = exp.run(i,j)

                        
                        np.save(os.path.join(dirname,f"spk_id_lvl{i}_it{j}"), spk_id)
                        np.save(os.path.join(dirname,f"spk_t_lvl{i}_it{j}"), spk_t)
                        if stim_gen is None:
                            # need to generate and save only once, ugly this way
                            np.save(os.path.join(dirname,"stim"), stim)
                            stim_gen = stim

                        del stim, spk_id, spk_t
                        
            case _: raise ValueError("invalid experiment selected, check json")

        return data_path
