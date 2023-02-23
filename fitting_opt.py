from bat_can import bat_can
import numpy as np
from bat_can_init import initialize
import sys
import ruamel.yaml
from fitting import fitting
import os
import shutil
from scipy.optimize import differential_evolution as optim_fun

input_file = 'Li_PorousSep_Sulfur_Assary_2step'
input_path = 'inputs/Li_PorousSep_Sulfur_Assary_2step.yaml'
write_file = 'Li_PorousSep_Sulfur_Assary_2stepw'
write_path = 'inputs/Li_PorousSep_Sulfur_Assary_2stepw.yaml'

config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(open(input_path))
data = config

p0 = np.array([1e11, 1e3, 1e4, 1e4,
                1e1]) #3.94520912e8 [1e13, 1e6, 1e5, 1e5, 1e2]
bounds = np.array([[1e6, 1e16], [1e1, 1e12],
               [1e-1, 1e11], [1e-1, 1e11],
               [1e-4, 1e8]])
#bounds = [(0, 1e14), (0, 1e8), (0, 1e7), (0, 1e7),
#          (0, 1e4)]

def run_kinetics(p_opt):
    # modify then dump and run a new yaml file for modifications to reaction
    param_section1 = 'lithium-electrolyte-reactions'
    param_section2 = 'carbon-elyte-reactions'
    param_section3 = 'elyte-bulk-reactions'
    param_section4 = 'lithium-sulfide-edge-reactions'
    param_section5 = 'lithium-sulfide-elyte-reactions'
    param_field1 = [] #['Li(b) <=> Li+(e) + electron']
    param_field2 = [] #['S8(e) + 2 electron <=> S8-(e)',
                    #'Li2S8(e) + 2 electron <=> Li2S8-(e)',
                    #'Li2S6(e) + 2 electron <=> Li2S6-(e)',
                    #'Li2S4(e) + 2 electron <=> Li2S4-(e)']
    param_field3 = ['S8-(e) + 2 Li+(e) <=> Li2S8(e)',
                    'Li2S8-(e) + 2 Li+(e) <=> Li2S2(e) + Li2S6(e)',
                    'Li2S8-(e) + 2 Li+(e) <=> 2 Li2S4(e)',
                    'Li2S6-(e) + 2 Li+(e) <=> Li2S2(e) + Li2S4(e)',
                    '2 Li2S3(e) => 2 Li2S2(e) + 0.25 S8(e)']
    param_field4 = [] #['Li2S4-(e) + 2 Li+(e) <=> Li2S(s) + Li2S3(e)']
    param_field5 = [] #['2 Li2S2(e) => 2 Li2S(s) + 0.25 S8(e)']
    param_name = 'rate-constant'
    param_coeff = 'A'
    #p_opt *= np.array([1e13, 1e3, 1e5, 1e5,
    #                1e2])
    param_list = [float(p) for p in p_opt]
    param_list1 = []#[param_list[0]]#param_list[0]
    param_list2 = []#param_list[1:5]
    param_list3 = param_list[0:5]
    param_list4 = [] #[param_list[5]]
    param_list5 = [] #[param_list[6]]
    SSR_0 = 1 #0.5555538510112178
    print('========================================================')
    print(p_opt)
    for param_i, param_new in enumerate(param_list1):
        #print('----------------------------------------------------')
        #print('Running for rate constant of reaction', param_field1[param_i])

        for elem in data[param_section1]:
            if elem['equation'] == param_field1[param_i]:
                elem[param_name][param_coeff] = param_new

        yaml = ruamel.yaml.YAML()
        yaml.indent(mapping=ind, sequence=ind, offset=bsi)
        with open(write_path, 'w') as file:
            yaml.dump(data, file)

    for param_i, param_new in enumerate(param_list2):
        #print('----------------------------------------------------')
        #print('Running for rate constant of reaction', param_field2[param_i])

        for elem in data[param_section2]:
            if elem['equation'] == param_field2[param_i]:
                elem[param_name][param_coeff] = param_new

        yaml = ruamel.yaml.YAML()
        yaml.indent(mapping=ind, sequence=ind, offset=bsi)
        with open(write_path, 'w') as file:
            yaml.dump(data, file)

    for param_i, param_new in enumerate(param_list3):
        #print('----------------------------------------------------')
        #print('Running for rate constant of reaction', param_field3[param_i])

        for elem in data[param_section3]:
            if elem['equation'] == param_field3[param_i]:
                elem[param_name][param_coeff] = param_new

        yaml = ruamel.yaml.YAML()
        yaml.indent(mapping=ind, sequence=ind, offset=bsi)
        with open(write_path, 'w') as file:
            yaml.dump(data, file)

    for param_i, param_new in enumerate(param_list4):
        #print('----------------------------------------------------')
        #print('Running for rate constant of reaction', param_field4[param_i])

        for elem in data[param_section4]:
            if elem['equation'] == param_field4[param_i]:
                elem[param_name][param_coeff] = param_new

        yaml = ruamel.yaml.YAML()
        yaml.indent(mapping=ind, sequence=ind, offset=bsi)
        with open(write_path, 'w') as file:
            yaml.dump(data, file)

    for param_i, param_new in enumerate(param_list5):
        #print('----------------------------------------------------')
        #print('Running for rate constant of reaction', param_field5[param_i])

        for elem in data[param_section5]:
            if elem['equation'] == param_field5[param_i]:
                elem[param_name][param_coeff] = param_new

        yaml = ruamel.yaml.YAML()
        yaml.indent(mapping=ind, sequence=ind, offset=bsi)
        with open(write_path, 'w') as file:
            yaml.dump(data, file)

    try:
        bat_can(write_file, 1, None)
        SSR = fitting(write_file)/SSR_0
    except:
        print('A parameter caused an error in the model')
        SSR = 1000*fitting(write_file)

    #SSR = fitting(write_file)
    print('For A values =', p_opt)
    print('SSR =', SSR)
    return SSR

opt_dict = {}
opt_dict['disp'] = True
#opt_dict['f_tol'] = 1e-12
#opt_dict['eps'] = 1e-8
#opt_dict['approx_grad'] = True
#res = minimize(run_kinetics, p0, method='Powell', bounds=b0, options=opt_dict)
res = optim_fun(run_kinetics, bounds, x0=p0) #, method='L-BFGS-B', bounds=b0, options=opt_dict)

print(res)
