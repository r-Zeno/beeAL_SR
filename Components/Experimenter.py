import numpy as np
import os
import time
from ExperimentStatic import *
from ExperimentDynamicSingle import *

class Experimenter:

    def __init__(self, model, exp_paras, folder, noise_lvls, spk_rec_steps, debugmode:bool):

        self.data_paths = []


    for run in noise_lvls:

