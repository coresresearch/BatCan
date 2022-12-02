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

param_opt_dict = {}

def run_kinetics(param_section, param_field, param_list, param_opt_dict):
    # modify then dump and run a new yaml file for modifications to reaction params
    source_dir = 'C:/users/korff/research/BatCan2/outputs/Fitting/' + write_file
    target_dir = 'C:/users/korff/research/BatCan2/outputs/Fitting/finished'

    #param_section = 'carbon-elyte-reactions'
    #param_field = 'S8(e) + 2 electron <=> S8-(e)'
    param_name = 'rate-constant'
    param_coeff = 'A'
    param_list = [int(param) for param in param_list]
    bad_params = []

    print('========================================================')
    for param_i, param_new in enumerate(param_list):
        print('----------------------------------------------------')
        print('Running batch for rate constant of reaction', param_field)
        print('Running parameter ', param_i+1, 'of', len(param_list))
        for elem in data[param_section]:
            if elem['equation'] == param_field:
                elem[param_name][param_coeff] = param_new

        yaml = ruamel.yaml.YAML()
        yaml.indent(mapping=ind, sequence=ind, offset=bsi)
        with open(write_path, 'w') as file:
            yaml.dump(data, file)

        try:
            bat_can(write_file, 1, None)
        except:
            print('Parameter ', str(param_i+1), 'caused an error in the model')
            bad_params.append(param_i)

    SSR_min, SSR_dict = fitting(write_file)

    for i, char in enumerate(SSR_min):
        if char.isdigit():
            int_i = i
            break

    param_opt_i = int(SSR_min[int_i:])

    if bad_params != []:
        for val in bad_params:
            if param_opt_i-1 >= val:
                param_opt_i += 1

    param_opt = param_list[param_opt_i-1]

    param_opt_dict[param_field] = 'parameter ' + str(param_opt_i)
    print(param_opt_dict[param_field])

    rxn_str = [elem['id'] for elem in data[param_section] if elem['equation'] == param_field]
    take_dir = os.listdir(source_dir)
    dest = os.path.join(target_dir, param_section, rxn_str[0])

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

    return param_opt_dict

A = 1e13
coeff_mult = np.linspace(0, 1, 2)
coeff_list = [1e12, 1e130, 1e14] #A*10**coeff_mult
run_kinetics('carbon-elyte-reactions', 'S8(e) + 2 electron <=> S8-(e)',
                coeff_list, param_opt_dict)

A = 1e10
coeff_list = A*10**coeff_mult
run_kinetics('carbon-elyte-reactions', 'Li2S8(e) + 2 electron <=> Li2S8-(e)',
                coeff_list, param_opt_dict)

print(param_opt_dict)

"=============================================================================="
"=============================================================================="
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
