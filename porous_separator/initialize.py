"""
    porous_separator.py

    Class file for porous separator methods
"""

import cantera as ct
import numpy as np

# Initialize the model.
def initialize(input_file, inputs, params, offset):
    class separator:
        elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])

        # State variables: electrolyte potential, electrolyte composition (nsp)
        nVars = 1 + elyte_obj.n_species

    # Set Cantera object state:
    if 'X_0' in inputs:
        separator.elyte_obj.TPX = params['T'], params['P'], inputs['X_0']
    else:
        separator.elyte_obj.TP = params['T'], params['P']

    separator.elyte_obj.electric_potential = inputs['phi_0']


    SV = np.zeros([separator.nVars])
    
    # Set up pointers:
    separator.SVptr = {}
    separator.SVptr['phi'] = 0
    separator.SVptr['C_k_elyte'] = np.arange(1, 
        1 + separator.elyte_obj.n_species)
    separator.SVptr['residual'] = np.arange(offset, offset+separator.nVars)

    # Load intial state variables:
    SV[separator.SVptr['phi']] = inputs['phi_0']
    SV[separator.SVptr['C_k_elyte']] = separator.elyte_obj.concentrations

    return SV, separator