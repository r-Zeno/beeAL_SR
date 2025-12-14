import numpy as np
import os
import time
from pygenn.genn_model import GeNNModel
from ExperimentStatic import *
from ExperimentDynamicSingle import *

class Experimenter:

    def __init__(self, model:GeNNModel, exp_paras:dict, folder:str, which_exp:str, debugmode:bool):

        self.data_paths = []
        self.folder = folder
        self.model = model
        self.debug = debugmode

        self.exp_type = which_exp
        self.paras = exp_paras[self.exp_type]
        # assuming all possible exp define runs and trial numbers
        self.runs = self.paras["num_runs"]
        self.trials = self.paras["num_trials"]

        self.path = os.path.join(folder, self.exp_type)

    def run(self):

        spk_id_paths = []
        spk_t_paths = []

        match self.exp_type:

            case "static_single":
                # work in progress
                # to move everything here
                raise ValueError("not ready yet")

            case "DynamicSingle":

                stim_path = []
                pop2record = self.paras["pop_to_record"]
                stim_gen = None
                data_log = []

                for i in range(self.runs):

                    dirname = os.path.join(self.folder,f"lvl_{i}")
                    os.makedirs(dirname, exist_ok=False)

                    for j in range(self.trials):

                        exp = ExperimentDynamicSingle(self.paras, self.model, stim_gen)
                        stim, spk_id, spk_t = exp.run(i)
                        
                        for pop in pop2record:

                            spk_id_path = os.path.join(dirname, f"spk_id_lvl{i}_it{j}_{pop}.npy")
                            spk_t_path = os.path.join(dirname, f"spk_t_lvl{i}_it{j}_{pop}.npy")
                            np.save(spk_id_path, spk_id[pop])
                            np.save(spk_t_path, spk_t[pop])

                            data_log.append({
                                "level": i,
                                "trial": j,
                                "pop": pop,
                                "spk_id_path": spk_id_path,
                                "spk_t_path": spk_t_path
                            })

                        if stim_gen is None:
                            # need to generate and save only once, ugly this way
                            stim_path = os.path.join(dirname, "stim.npy")
                            np.save(stim_path, stim)

                            stim_gen = stim

                        del stim, data_log
                        
            case _: raise ValueError("invalid experiment selected, check json")

        return stim_path, spk_id_paths, spk_t_paths
