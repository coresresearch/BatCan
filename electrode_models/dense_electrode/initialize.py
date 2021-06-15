"""
    single_particle_electrode.py

    Class file for dense metal (e.g. Li) electrode methods
"""

import cantera as ct
import numpy as np

# Initialize the model.
def initialize(input_file, inputs, sep_inputs, counter_inputs, electrode_name, 
        params, offset):
    
    class electrode:
        bulk_obj = ct.Solution(input_file, inputs['bulk-phase'])
        elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])
        conductor_obj = ct.Solution(input_file, inputs['conductor-phase'])
        surf_obj = ct.Interface(input_file, inputs['surf-phase'], 
            [bulk_obj, elyte_obj, conductor_obj])
        name = electrode_name
        if name=='anode':
            i_ext_flag = -1
        elif name=='cathode':
            i_ext_flag = 1
        else:
            raise ValueError("Electrode must be an anode or a cathode.")

        index_Li = elyte_obj.species_index(inputs['mobile-ion'])

        dy = inputs['thickness']
        min_thickness = inputs['minimum-thickness']

        A_surf_ratio = inputs['A_surf_ratio']
        C_dl_Inv = 1/inputs['C_dl']

        # Microstructure-based transport scaling factor, based on Bruggeman 
        # coefficient of -0.5:
        dy_elyte = inputs['dy_elyte']
        eps_elyte = sep_inputs['eps_electrolyte']
        elyte_microstructure = eps_elyte**1.5
        
        SV_offset = offset
        
        # Li anode is not capacity-limiting, in and of itself.  Rather the 
        # total amount of Li in the system is the limit. This is done in a 
        # separate routine, at a later time.
        capacity = 1e21
            
        # State variables: electrode potential, electrolyte potential, 
        # thickness, electrolyte composition (nsp)
        nVars = 3 + elyte_obj.n_species

        # Load the residual model and store it as a method of this class:
        from .functions import residual, make_alg_consistent, voltage_lim


    # Set Cantera object state. If the user provided an initial composition, 
    # use that, here:
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
    electrode.SVptr['phi_ed'] = np.array([0])
    electrode.SVptr['phi_dl'] = np.array([1])
    electrode.SVptr['thickness'] = np.array([2])
    electrode.SVptr['C_k_elyte'] = np.arange(3, 
            3 + electrode.elyte_obj.n_species)

    electrode.SVptr['residual'] = np.arange(offset, offset+electrode.nVars)

    # Save the indices of any algebraic variables:
    electrode.algvars = offset + electrode.SVptr['phi_ed'][:]
    
    # Load intial state variables:
    SV[electrode.SVptr['phi_ed']] = inputs['phi_0']
    SV[electrode.SVptr['phi_dl']] = sep_inputs['phi_0'] - inputs['phi_0']
    SV[electrode.SVptr['thickness']] = inputs['thickness']
    SV[electrode.SVptr['C_k_elyte']] = electrode.elyte_obj.concentrations

    return SV, electrode
