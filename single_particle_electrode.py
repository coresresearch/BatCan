"""
    single_particle_electrode.py

    Class file for single-particle electrode methods
"""

import cantera as ct
import numpy as np

# Initialize the model.
def initialize(input_file, inputs, electrode_name, phi_elyte_0, params):
    class electrode:
        bulk_obj = ct.Solution(input_file, inputs['bulk-phase'])
        elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])
        conductor_obj = ct.Solution(input_file, inputs['conductor-phase'])
        surf_obj = ct.Interface(input_file, inputs['surf-phase'], 
            [bulk_obj, elyte_obj, conductor_obj])
        name = electrode_name

    # Set Cantera object state:
    if 'X_0' in inputs:
        electrode.bulk_obj.TPX = params['T'], params['P'], inputs['X_0']
    else:
        electrode.bulk_obj.TP = params['T'], params['P']

    electrode.bulk_obj.electric_potential = inputs['phi_0']

    # State variables: electrode potential, electrolyte potential, electrode 
    #    composition (nsp), electrolyte composition (nsp)
    nVars = 2 + electrode.bulk_obj.n_species + electrode.elyte_obj.n_species
    SV = np.zeros([nVars])

    # Set up pointers:
    SVptr = {}
    SVptr['phi_ed'] = 0
    SVptr['phi_dl'] = 1
    SVptr['C_k_ed'] = np.arange(2, 2 + electrode.bulk_obj.n_species)
    SVptr['C_k_elyte'] = np.arange(2 + electrode.bulk_obj.n_species, 
        2 + electrode.bulk_obj.n_species + electrode.elyte_obj.n_species)

    # Load intial state variables:
    SV[SVptr['phi_ed']] = inputs['phi_0']
    SV[SVptr['phi_dl']] = phi_elyte_0 - inputs['phi_0']
    SV[SVptr['C_k_ed']] = electrode.bulk_obj.concentrations
    SV[SVptr['C_k_elyte']] = electrode.elyte_obj.concentrations

    return SV, electrode

