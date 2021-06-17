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

        from .functions import residual, electrode_boundary_flux
    
        dy = inputs['thickness']
        dyInv = 1/dy
        eps_elyte = inputs['eps_electrolyte']

        # Microstructure-based transport scaling factor, based on Bruggeman 
        # coefficient of -0.5:
        elyte_microstructure = eps_elyte**1.5

        # Ionic conductivity of bulk electrolyte (S/m):
        sigma_io = inputs['sigma_io']

    SV = np.zeros([separator.nVars])
    
    # Set up pointers:
    separator.SVptr = {}
    separator.SVptr['phi'] = 0
    separator.SVptr['residual'] = np.arange(offset, offset+separator.nVars)

    # Save indices for any algebraic variables.
    separator.algvars = [offset + separator.SVptr['phi']]
    
    # Load intial state variables:
    SV[separator.SVptr['phi']] = inputs['phi_0']

    return SV, separator