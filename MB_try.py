import json
from ModelBuilder import ModelBuilder

with open('/Users/zenorossi/beeAL/example_for_modelbuilds.json') as f:
    parameters = json.load(f)
    print(parameters)

builder = ModelBuilder(parameters)
model = builder.build()
