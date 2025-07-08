import json
from ModelBuilder import ModelBuilder
from Experimenter import Experimenter
from Analyzer import Analyzer

with open('/Users/zenorossi/beeAL/example_for_modelbuilds.json') as f:
    mod_parameters = json.load(f)

with open('/Users/zenorossi/beeAL/exp_paras.json') as f:
    exp_parameters = json.load(f)

with open('/Users/zenorossi/beeAL/analysis_params.json') as f:
    an_parameters = json.load(f)

builder = ModelBuilder(mod_parameters)
model = builder.build(3.0)

experiment = Experimenter(model, experiment_parameters=exp_parameters)
data_folder = experiment.run()

analysis = Analyzer(data_folder, an_parameters, mod_parameters)
analysis.analyze()
