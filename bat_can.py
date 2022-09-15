# bat_can.py
"""
    BatCan - Battery Modeling in Cantera

    This file reads in the user input, runs the simulation, and then produces any requested output (saved data file, preliminary plots, etc.)

"""
# Import modules
from datetime import datetime
import importlib # allows us to import from user input string.
#import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import os
from shutil import copy2
import timeit
#import model_run_def

from bat_can_init import initialize

# This is the main function that runs the model.  We define it this way so it
# is called by "main," below:
def bat_can(input, cores, n_ca):
    # Record the start time:
    start = timeit.default_timer()

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

    if not cores:
        cores = 1

    if not n_ca:
        n_ca = 0
    #===========================================================================
    #   READ IN USER INPUTS
    #===========================================================================
    an_inputs, sep_inputs, ca_inputs, parameters = initialize(input_file)

    now = datetime.now()
    dt =  now.strftime("%Y%m%d_%H%M")

    # Save name of input file, without path or extension:
    parameters['input'] = input
    parameters['output'] = 'outputs/' + parameters['input']+ '_' + dt
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
    if n_ca != 0:
        ca_inputs['n-points'] = n_ca
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

    # If the user requests a specific initailization routine, run that first:
    if 'initialize' in parameters and parameters['initialize']['enable']:
        if parameters['initialize']['type'] == 'open-circuit':
            model = importlib.import_module('.'+'CC_cycle',
                package='simulations')

            t_span = parameters['initialize']['time']

            sim = {'i_ext': None, 'C-rate': 0., 'n_cycles': 0,
                'first-step': 'discharge', 'equilibrate':
                {'enable': True, 'time':  t_span}, 'phi-cutoff-lower': 2.0,
                'phi-cutoff-upper': 4.8, 'init':True}

            solution = model.run(SV_0, an, sep, ca, algvars, parameters, sim)

            # Save final state as the initial state for all subsequent
            # simulation steps:
            SV_0 = model.final_state(solution)

        else:
            raise ValueError("Initialization method currently not implemented.")

    global model_run
    def model_run(sim):

        # Import the simulation to be run:
        model = importlib.import_module('.'+sim['type'], package='simulations')

        sim['init'] = False
        #ca_inputs['n_points'] = sim['n_ca']
        #ca_module = importlib.import_module('electrode_models.'
        #    + ca_inputs['class'])
        #ca = ca_module.electrode(input_file, ca_inputs, sep_inputs, an_inputs,
        #    'cathode', parameters, offset= an.n_vars+sep.n_vars*sep.n_points)
        # Run the simulation
        solution = model.run(SV_0, an, sep, ca, algvars, parameters, sim)

        # Call any output routines related to the simulation type:
        model.output(solution, an, sep, ca, parameters, sim)

        SV_init = model.initial_state(solution)
        return SV_init

    #=======================================================================
    #   CREATE FIGURES AND SAVE ALL OUTPUTS
    #=======================================================================

    # If the user specified to use multiple cores (only relevant if there are
    # multiple simulations), run them in a multiprocessing pool:
    #sim = parameters['simulations']
    pool = Pool(int(cores))
    SV_0 = pool.map(model_run, list(parameters['simulations']))

    if len(parameters['simulations']) == 1 and not parameters['cell-test']['enable']:
        filename = (parameters['output'] +'_'
                    + parameters['simulations'][0]['outputs']['save-name'] )
    else:
        filename = (parameters['output'] +'/')

    if not os.path.exists(filename):
        os.makedirs(filename)

    copy2(input_file, filename)

    # Record time when finished:
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
    parser.add_argument('--cores')
    parser.add_argument('--n_ca')
    args = parser.parse_args()

    bat_can(args.input, args.cores, args.n_ca)
