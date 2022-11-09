from bat_can import bat_can
import numpy as np
from bat_can_init import initialize
import sys
import ruamel.yaml

input_file = 'Li_PorousSep_Sulfur_Assary1'
input_path = 'inputs/Li_PorousSep_Sulfur_Assary1.yaml'
write_file = 'Li_PorousSep_Sulfur_Assary1w'
write_path = 'inputs/Li_PorousSep_Sulfur_Assary1w.yaml'

config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(open(input_path))
data = config
"=============================================================================="
# modify and then dump to a new yaml file for modifications to the cell
#   description
param_section = 'cell-description'
param_field = 'cathode'
param_name = 'n-points'
param_list = []

for param_new in param_list:
    for elem in data[param_section]:
        if elem == param_field:
            data[param_section][elem][param_name] = param_new

    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=ind, sequence=ind, offset=bsi)
    with open(write_path, 'w') as file:
        yaml.dump(data, file)

    bat_can(write_file, 1, None)

"=============================================================================="
# modify then dump and run a new yaml file for modifications to simulation input
param_section = 'parameters'
param_field = 'simulations'
param_name = 'C-rate'
param_list = []

for param_new in param_list:
    for elem in data[param_section]:
        if elem == param_field:
            data[param_section][elem][0][param_name] = param_new

    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=ind, sequence=ind, offset=bsi)
    with open(write_path, 'w') as file:
        yaml.dump(data, file)

    bat_can(write_file, 1, None)

"=============================================================================="
# modify then dump and run a new yaml file for modifications to species params
param_section = 'species'
param_field = 'Li+(e)'
param_name = 'h0'
param_list = [] # string containing value of thermo param and units
                             #  (e.g. '-278 kJ/mol')

for param_new in param_list:
    for elem in data[param_section]:
        if elem['name'] == param_field:
            elem['thermo'][param_name] = param_new

    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=ind, sequence=ind, offset=bsi)
    with open(write_path, 'w') as file:
        yaml.dump(data, file)

    bat_can(write_file, 1, None)

"=============================================================================="
# modify then dump and run a new yaml file for modifications to reaction params
param_section = 'elyte-bulk-reactions'
param_field = 'Li2S8(e) => Li2S6(e) + 0.25 S8(e)'
param_name = 'rate-constant'
param_coeff = 'A'
param_list = [1e-3, 2e-3, 3e-3]
if param_list != []:
    print('Running batch for rate constant of reaction', param_field)
for param_new in param_list:
    for elem in data[param_section]:
        if elem['equation'] == param_field:
            elem[param_name][param_coeff] = param_new

    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=ind, sequence=ind, offset=bsi)
    with open(write_path, 'w') as file:
        yaml.dump(data, file)

    bat_can(write_file, 1, None)
