"""
    porous_separator.py

    Class file for porous separator methods
"""

import cantera as ct
import numpy as np

# Initialize the model.
def initialize(input_file, inputs, params, offset):
    class separator:
        # State variables: electrolyte potential
        nVars = 1

        from .residual import residual
    
    SV = np.zeros([separator.nVars])
    
    # Set up pointers:
    separator.SVptr = {}
    separator.SVptr['phi'] = 0
    separator.SVptr['residual'] = np.arange(offset, offset+separator.nVars)

    # Load intial state variables:
    SV[separator.SVptr['phi']] = inputs['phi_0']

    return SV, separator