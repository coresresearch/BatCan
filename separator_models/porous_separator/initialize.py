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

        from .functions import (residual, electrode_boundary_flux, 
            make_alg_consistent)
    
        n_points = inputs['n_points']
        dy = inputs['thickness']/n_points
        dyInv = 1/dy
        eps_elyte = inputs['eps_electrolyte']

        # Microstructure-based transport scaling factor, based on Bruggeman 
        # coefficient of -0.5:
        elyte_microstructure = eps_elyte**1.5

        # Ionic conductivity of bulk electrolyte (S/m):
        sigma_io = inputs['sigma_io']

    # Set Cantera object state:
    if 'X_0' in inputs:
        separator.elyte_obj.TPX = params['T'], params['P'], inputs['X_0']
    else:
        separator.elyte_obj.TP = params['T'], params['P']

    separator.elyte_obj.electric_potential = inputs['phi_0']

    SV = np.zeros([separator.n_points*separator.nVars])
    
    # Set up pointers:
    separator.SVptr = {}
    separator.SVptr['phi'] = np.arange(0,separator.n_points*separator.nVars,
        separator.nVars, dtype='int')

    separator.SVptr['C_k elyte'] = np.ndarray(shape=(separator.n_points, 
        separator.elyte_obj.n_species), dtype='int')       
    for i in range(separator.n_points):
        separator.SVptr['C_k elyte'][i,:] = range(1 + i*separator.nVars, 
            1 + i*separator.nVars + separator.elyte_obj.n_species)
    
    # What portion of the SV represents the separator?
    separator.SVptr['residual'] = np.arange(offset, 
        offset+separator.n_points*separator.nVars)

    # Save indices for any algebraic variables.
    separator.algvars = offset + separator.SVptr['phi']

    # Load intial state variables:
    SV[separator.SVptr['phi']] = inputs['phi_0']
    for i in range(separator.n_points):
        SV[separator.SVptr['C_k elyte'][i,:]] = \
            separator.elyte_obj.concentrations

    return SV, separator