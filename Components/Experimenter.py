import numpy as np
import os
import time
from ExperimentStatic import *
from ExperimentDynamicSingle import *

class Experimenter:

    def __init__(self, model, exp_paras, folder, noise_lvls, spk_rec_steps, debugmode:bool):

        self.data_paths = []
        self.which_exp = exp_paras["exp"]
        

    def run():

        match self.which_exp:
            case "static_single":
                exp = ExperimentStatic()
                exp.run()
            case "dynamic_single":
                exp = 0
            case _: raise ValueError("invalid experiment selected, check json")
                

