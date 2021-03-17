"""
    porous_separator.py

    Class file for porous separator methods
"""

import cantera as ct
import numpy as np

# Initialize the model.
def initialize(input_file, inputs, params):
    class separator:
        elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])

    # Set Cantera object state:
    if 'X_0' in inputs:
        separator.elyte_obj.TPX = params['T'], params['P'], inputs['X_0']
    else:
        separator.elyte_obj.TP = params['T'], params['P']

    separator.elyte_obj.electric_potential = inputs['phi_0']

    # State variables: electrolyte potential, electrolyte composition (nsp)
    nVars = 1 + separator.elyte_obj.n_species
    SV = np.zeros([nVars])
    
    # Set up pointers:
    SVptr = {}
    SVptr['phi'] = 0
    SVptr['C_k_elyte'] = np.arange(1, 1 + separator.elyte_obj.n_species)

    # Load intial state variables:
    SV[SVptr['phi']] = inputs['phi_0']
    SV[SVptr['C_k_elyte']] = separator.elyte_obj.concentrations

    return SV, separator