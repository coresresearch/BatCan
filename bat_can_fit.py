# bat_can.py
"""
    BatCan - Battery Modeling in Cantera

    This file reads in the user input, runs the simulation, and then produces any requested output (saved data file, preliminary plots, etc.)

"""
# Import modules
import importlib # allows us to import from user input string.
import numpy as np
import pandas as pd
from scipy import optimize as spo
import timeit

from submodels.fitting import SSR_voltage_capacity as SSR
from bat_can_init import initialize

# This is the main function that runs the model.  We define it this way so it 
# is called by "main," below:
def bat_can(input = None, return_type = None):
    if input is None:
        # Default is a single-particle model of graphite/LCO
        input_file = 'inputs/spmGraphite_PorousSep_spmLCO_input.yaml'
    else:
        if input[-5:] == '.yaml':
            input_file  = 'inputs/'+input

            # Strip the file extension:
            input = input[:-4]
        else:
            input_file = 'inputs/'+input+'.yaml'

    #===========================================================================
    #   READ IN USER INPUTS
    #===========================================================================
    an_inputs, sep_inputs, ca_inputs, parameters, fit_params = \
        initialize(input_file)

    # Save name of input file, without path or extension:
    parameters['input'] = input
    #===========================================================================
    #   CREATE ELEMENT CLASSES AND INITIAL SOLUTION VECTOR SV_0
    #===========================================================================
    # For each element (anode 'an', separator 'sep', cathode 'ca') the 'class' 
    # variable from the inputs tells what kind of anode, separator, or cathode 
    # it is, and points to a '.py' file in this directory.  We import that 
    # module, and then run its 'initialize' routine to create an intial 
    # solution vector and an object that stores needed parameters.
    # import single_particle_electrode as an_module_0
    an_module = importlib.import_module('electrode_models.' 
        + an_inputs['class'])
    an = an_module.electrode(input_file, an_inputs, sep_inputs, ca_inputs, 
            'anode', parameters, offset=0)
    
    sep_module = importlib.import_module('separator_models.' 
        + sep_inputs['class'])
    sep = sep_module.separator(input_file, sep_inputs, parameters, 
            offset=an.n_vars)
    
    # Check to see if the anode object needs to adjust the separator properties:
    sep = an.adjust_separator(sep)
    ca_module = importlib.import_module('electrode_models.' 
        + ca_inputs['class'])
    ca = ca_module.electrode(input_file, ca_inputs, sep_inputs, an_inputs, 
        'cathode', parameters, offset= an.n_vars+sep.n_vars*sep.n_points)

    # Check to see if the cathode object needs to adjust the separator 
    # properties:    
    sep = ca.adjust_separator(sep)

    # Initialize the solution vector:
    SV_an_0 = an.initialize(an_inputs, sep_inputs)
    SV_sep_0 = sep.initialize(sep_inputs)
    SV_ca_0 = ca.initialize(ca_inputs, sep_inputs)

    # Stack the three initial solution vectors into a single vector:
    SV_0 = np.hstack([SV_an_0, SV_sep_0, SV_ca_0])
    # Ditto for the algebraic variable indices:
    algvars = np.hstack([an.algvars, sep.algvars, ca.algvars])

    #===========================================================================
    #   RUN THE SIMULATION
    #===========================================================================
    # The inputs tell us what type of experiment we will simulate.  Load the 
    # module, then call its 'run' function:
    if 'initialize' in parameters and parameters['initialize']['enable']:
        if parameters['initialize']['type'] == 'open-circuit':
            model = importlib.import_module('.'+'CC_cycle', 
                package='simulations')

            sim = {'i_ext': None, 'C-rate': 0.0, 'n_cycles': 0, 
                'first-step': 'discharge', 'equilibrate': 
                {'enable': True, 'time':  5}, 'phi-cutoff-lower': 2.0, 
                'phi-cutoff-upper': 4.8}
            
            solution = model.run(SV_0, an, sep, ca, algvars, parameters, sim)
            
            # Save final state as the initial state for all subsequent 
            # simulation steps:
            SV_0 = model.final_state(solution)

        else:
            raise ValueError("Initialization method currently not implemented.")


    def run_model(x_guess):
        print('x = ',x_guess)
    
        # Right now we just do diffusion coefficients:
        for i, x in enumerate(x_guess):
            if fit_params[i]['type'] == 'elyte-transport':
                sep.D_k[sep.elyte_obj.species_index(fit_params[i]['species'])] \
                    = x
            if fit_params[i]['type'] == 'cathode-kinetics':
                ca.surf_obj.set_multiplier(x, fit_params[i]['reaction-index'])

        outcome = 0
        for sim in parameters['simulations']:
            try:
                model = importlib.import_module('.'+sim['type'], package='simulations')

                solution = model.run(SV_0, an, sep, ca, algvars, parameters, sim)

                # Read out the results:
                results = model.output(solution, an, sep, ca, parameters, sim, True)

                sim_data = np.array((results['capacity'].to_numpy(),
                    results['phi_ed'].iloc[:,-1]))
                
                ssr_calc = SSR(sim['ref_data'].to_numpy(), sim_data)
                print('SSR = ', ssr_calc)
                outcome += ssr_calc 
            except:
                # Assign a large penalty for failed parameter sets:
                ssr_calc = 1e23
                print('Simulation failed')
                outcome += ssr_calc

        print('SSR = ', outcome)
        return outcome

    start = timeit.default_timer()
    x_start = []
    x_bounds = []
    for x in fit_params:
        x_start.append(x['guess-value'])
        x_bounds.append((1e-15, None))

    for sim in parameters['simulations']:
        sim['ref_data'] = pd.read_excel('data/' + sim['validation'])

    # x_bounds = [(1e-15, None), (1e-15, None)]
    result = spo.minimize(run_model, x_start, bounds = x_bounds, 
        options={'disp': True})
    print(result)

    stop = timeit.default_timer()
    print('Time: ', stop - start)  

#===========================================================================
#   FUNCTIONALITY TO RUN FROM THE COMMAND LINE
#===========================================================================
if __name__ == '__main__':
    import argparse

    # Currently, the only command line keyword enabled is --input, to specify 
    # the input file location:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    args = parser.parse_args()
    
    bat_can(args.input)
