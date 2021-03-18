# bat_can.py
"""
    BatCan - Battery Modeling in Cantera

    This file reads in the user input, runs the simulation, and then produces any requested output (saved data file, preliminary plots, etc.)

"""
# Import modules
import importlib # allows us to import from user input string.
from bat_can_init import initialize as init
import numpy as np

def bat_can(input = None):
    if input is None:
        # Default is a single-particle model of graphite/LCO
        input = 'bat_can_input.yaml'

    #===========================================================================
    #   READ IN USER INPUTS
    #===========================================================================
    an_inputs, sep_inputs, ca_inputs, parameters = init(input)

    #===========================================================================
    #   CREATE ELEMENT CLASSES AND INITIAL SOLUTION VECTOR SV_0
    #===========================================================================
    # For each element (anode 'an', separator 'sep', cathode 'ca') the 'class' 
    # variable from the inputs tells what kind of anode, separator, or cathode 
    # it is, and points to a '.py' file in this directory.  We import that 
    # module, and then run its 'initialize' routine to create an intial 
    # solution vector and an object that stores needed parameters.
    an_module = importlib.import_module(an_inputs['class'])
    SV_an_0, anode =  an_module.initialize(input, an_inputs, 'anode', 
        sep_inputs['phi_0'], parameters)

    sep_module = importlib.import_module(sep_inputs['class'])
    SV_sep_0, separator = sep_module.initialize(input, sep_inputs, parameters)

    ca_module = importlib.import_module(ca_inputs['class'])
    SV_ca_0, cathode = ca_module.initialize(input, ca_inputs, 'cathode', 
        sep_inputs['phi_0'], parameters)

    # Stack the three initial solution vectors into a single vector:
    SV_0 = np.hstack([SV_an_0, SV_sep_0, SV_ca_0])

    #===========================================================================
    #   RUN THE MODEL
    #===========================================================================
    # The inputs tell us what type of experiment we will simulate.  Load the 
    # module, then call its 'run' function:
    model = importlib.import_module(parameters['simulation']['type'])

    t, solution = model.run(SV_0, anode, separator, cathode, parameters)
    """TEST"""
    # Should be zero:
    print('Max difference = ', max(solution[-1,:]-SV_0))

    #===========================================================================
    #   CREATE FIGURES AND SAVE ALL OUTPUTS
    #===========================================================================
    # Call any output routines related to the simulation type:
    model.output(solution)


#===========================================================================
#   FUNCTIONALITY TO RUN FROM THE COMMAND LINE
#===========================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    args = parser.parse_args()
    
    bat_can(args.input)