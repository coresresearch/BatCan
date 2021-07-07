# bat_can.py
"""
    BatCan - Battery Modeling in Cantera

    This file reads in the user input, runs the simulation, and then produces any requested output (saved data file, preliminary plots, etc.)

"""
# Import modules
import importlib # allows us to import from user input string.
import numpy as np

from bat_can_init import initialize

# This is the main function that runs the model.  We define it this way so it 
# is called by "main," below:
def bat_can(input = None):
    if input is None:
        # Default is a single-particle model of graphite/LCO
        input = 'inputs/spm_Graphite_Porous_Sep_spm_LCO_input.yaml'
    else:
        if input[-5:] == '.yaml':
            input  = 'inputs/'+input
        else:
            input = 'inputs/'+input+'.yaml'

    #===========================================================================
    #   READ IN USER INPUTS
    #===========================================================================
    an_inputs, sep_inputs, ca_inputs, parameters = initialize(input)

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
    an = an_module.electrode(input, an_inputs, sep_inputs, ca_inputs, 
            'anode', parameters, offset=0)
    SV_an_0 = an.initialize(an_inputs, sep_inputs)
    
    sep_module = importlib.import_module('separator_models.' 
        + sep_inputs['class'])
    sep = sep_module.separator(input, sep_inputs, parameters, 
            offset=an.n_vars)
    SV_sep_0 = sep.initialize(sep_inputs)

    ca_module = importlib.import_module('electrode_models.' 
        + ca_inputs['class'])
    ca = ca_module.electrode(input, ca_inputs, sep_inputs, an_inputs, 
            'cathode', parameters, offset=sep.SVptr['sep'][-1]+1)
    SV_ca_0 = ca.initialize(ca_inputs, sep_inputs)

    # Check to see if either of the electrode objects need to adjust the 
    # separator properties:
    sep = an.adjust_separator(sep)
    sep = ca.adjust_separator(sep)

    # Stack the three initial solution vectors into a single vector:
    SV_0 = np.hstack([SV_an_0, SV_sep_0, SV_ca_0])
    # Ditto for the algebraic variable indices:
    algvars = np.hstack([an.algvars, sep.algvars, ca.algvars])

    #===========================================================================
    #   RUN THE SIMULATION
    #===========================================================================
    # The inputs tell us what type of experiment we will simulate.  Load the 
    # module, then call its 'run' function:
    model = importlib.import_module('.'+parameters['simulation']['type'], 
            package='simulations')

    solution = model.run(SV_0, an, sep, ca, algvars, parameters)

    #===========================================================================
    #   CREATE FIGURES AND SAVE ALL OUTPUTS
    #===========================================================================
    # Call any output routines related to the simulation type:
    model.output(solution, an, sep, ca, parameters)


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
