"""
    single_particle_electrode.py

    Class file for single-particle electrode methods
"""

import cantera as ct
import numpy as np

# Initialize the model.
def initialize(input_file, inputs, electrode_name, phi_elyte_0, params, offset):
    class electrode:
        bulk_obj = ct.Solution(input_file, inputs['bulk-phase'])
        elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])
        conductor_obj = ct.Solution(input_file, inputs['conductor-phase'])
        surf_obj = ct.Interface(input_file, inputs['surf-phase'], 
            [bulk_obj, elyte_obj, conductor_obj])
        name = electrode_name
        if name=='anode':
            i_ext_flag = 1
        elif name=='cathode':
            i_ext_flag = -1
        else:
            raise ValueError("Electrode must be an anode or a cathode.")

        A_surf_ratio = (3*inputs['eps_solid']*inputs['thickness']/inputs['r_p'])
        C_dl_Inv = 1/inputs['C_dl']

        SV_offset = offset
            
        # State variables: electrode potential, electrolyte potential, 
        # electrode composition (nsp), electrolyte composition (nsp)
        nVars = 2 + bulk_obj.n_species + elyte_obj.n_species

        # Load the residual model and store it as a method of this class:
        from .residual import residual


    # Set Cantera object state:
    if 'X_0' in inputs:
        electrode.bulk_obj.TPX = params['T'], params['P'], inputs['X_0']
    else:
        electrode.bulk_obj.TP = params['T'], params['P']

    electrode.elyte_obj.TP = params['T'], params['P']
    electrode.surf_obj.TP = params['T'], params['P']
    electrode.conductor_obj.TP = params['T'], params['P']
    electrode.bulk_obj.electric_potential = inputs['phi_0']

    SV = np.zeros([electrode.nVars])

    # Set up pointers:
    electrode.SVptr = {}
    electrode.SVptr['phi_ed'] = 0
    electrode.SVptr['phi_dl'] = 1
    electrode.SVptr['C_k_ed'] = np.arange(2, 2 + electrode.bulk_obj.n_species)
    electrode.SVptr['C_k_elyte'] = np.arange(2 + electrode.bulk_obj.n_species, 
        2 + electrode.bulk_obj.n_species + electrode.elyte_obj.n_species)

    electrode.SVptr['residual'] = np.arange(offset, offset+electrode.nVars)

    # Load intial state variables:
    SV[electrode.SVptr['phi_ed']] = inputs['phi_0']
    SV[electrode.SVptr['phi_dl']] = phi_elyte_0 - inputs['phi_0']
    SV[electrode.SVptr['C_k_ed']] = electrode.bulk_obj.concentrations
    SV[electrode.SVptr['C_k_elyte']] = electrode.elyte_obj.concentrations

    return SV, electrode
