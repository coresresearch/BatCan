# bat_can.py
"""
    BatCan - Battery Modeling in Cantera

    This file reads in the user input, runs the simulation, and then produces any requested output (saved data file, preliminary plots, etc.)

"""
# Import modules
import importlib # allows us to import from user input string.
from bat_can_init import initialize as init
import numpy as np
from scikits.odes.odeint import odeint

def bat_can(input = None):
    if input is None:
        # Default is a single-particle model of graphite/LCO
        input = 'bat_can_input.yaml'

    #===========================================================================
    #   READ IN USER INPUTS
    #===========================================================================
    an_inputs, sep_inputs, ca_inputs, params = init(input)

    #===========================================================================
    #   CREATE ELEMENT CLASSES AND INITIAL SOLUTION VECTOR
    #===========================================================================
    anode_module = importlib.import_module(an_inputs['class'])
    SV_an_0, anode =  anode_module.initialize(input, an_inputs, 'anode', 
        sep_inputs['phi_0'], params)

    separator_module = importlib.import_module(sep_inputs['class'])
    SV_sep_0, separator = separator_module.initialize(input, sep_inputs, params)

    cathode_module = importlib.import_module(ca_inputs['class'])
    SV_ca_0, cathode = cathode_module.initialize(input, ca_inputs, 'cathode', 
        sep_inputs['phi_0'], params)

    SV_0 = np.hstack([SV_an_0, SV_sep_0, SV_ca_0])


    """TEST"""
    # Integrate:
    t_out = np.linspace(0,1)
    def residual(t,SV, SVdot):
        resid = np.zeros_like(SV)
        SVdot = resid

    output = odeint(residual, t_out, SV_0)
    # Should be zero:
    print(max(output.values.y[-1,:]-SV_0))
# Run from command line:
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    args = parser.parse_args()
    
    bat_can(args.input)