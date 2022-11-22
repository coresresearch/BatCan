from bat_can import bat_can
import numpy as np
from bat_can_init import initialize
import sys
import ruamel.yaml
from fitting import fitting
import os
import shutil

input_file = 'Li_PorousSep_Sulfur_Assary_2step'
input_path = 'inputs/Li_PorousSep_Sulfur_Assary_2step.yaml'
write_file = 'Li_PorousSep_Sulfur_Assary_2stepw'
write_path = 'inputs/Li_PorousSep_Sulfur_Assary_2stepw.yaml'

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
source_dir = 'C:/users/korff/research/BatCan2/outputs/Fitting/' + write_file
target_dir = 'C:/users/korff/research/BatCan2/outputs/Fitting/finished'

param_section = 'carbon-elyte-reactions'
param_field = 'S8(e) + 2 electron <=> S8-(e)'
param_name = 'rate-constant'
param_coeff = 'A'
param_list = [1e12, 1e13]
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

SSR_min, SSR_dict = fitting(write_file)
param_opt = param_list[int(SSR_min[-1])-1]
print(param_opt)
take_dir = os.listdir(source_dir)

dest = os.path.join(target_dir, param_section, 'rxn1')

if not os.path.exists(dest):
    os.makedirs(dest)

for dir in take_dir:
    source = os.path.join(source_dir, dir)
    shutil.move(source, dest)

for elem in data[param_section]:
    if elem['equation'] == param_field:
        elem[param_name][param_coeff] = param_opt

yaml = ruamel.yaml.YAML()
yaml.indent(mapping=ind, sequence=ind, offset=bsi)
with open(write_path, 'w') as file:
    yaml.dump(data, file)
"------------------------------------------------------------------------------"
param_section = 'carbon-elyte-reactions'
param_field = 'Li2S8(e) + 2 electron <=> Li2S8-(e)'
param_name = 'rate-constant'
param_coeff = 'A'
param_list = [1e9, 1e10, 1e11]
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

SSR_min, SSR_dict = fitting(write_file)
param_opt = param_list[int(SSR_min[-1])-1]
print(param_opt)
take_dir = os.listdir(source_dir)

dest = os.path.join(target_dir, param_section, 'rxn2')

if not os.path.exists(dest):
    os.makedirs(dest)

for dir in take_dir:
    source = os.path.join(source_dir, dir)
    shutil.move(source, dest)

for elem in data[param_section]:
    if elem['equation'] == param_field:
        elem[param_name][param_coeff] = param_opt

yaml = ruamel.yaml.YAML()
yaml.indent(mapping=ind, sequence=ind, offset=bsi)
with open(write_path, 'w') as file:
    yaml.dump(data, file)
