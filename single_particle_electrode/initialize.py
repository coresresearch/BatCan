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
            i_ext_flag = -1
        elif name=='cathode':
            i_ext_flag = 1
        else:
            raise ValueError("Electrode must be an anode or a cathode.")

        index_Li = elyte_obj.species_index(inputs['mobile-ion'])

        dy = inputs['thickness']
        dyInv = 1/dy
        eps_solid = inputs['eps_solid']
        eps_elyte = 1 - eps_solid

        A_surf_ratio = (3*eps_solid*dy/inputs['r_p'])
        C_dl_Inv = 1/inputs['C_dl']

        # Microstructure-based transport scaling factor, based on Bruggeman 
        # coefficient of -0.5:
        elyte_microstructure = eps_elyte**1.5
        
        SV_offset = offset

        # Determine Capacity (Ah/m2)
        X_o = bulk_obj.X # save initial X
        bulk_obj.X = inputs['stored-ion']['name']+':1.0'
        C = bulk_obj[inputs['stored-ion']['name']].concentrations[0]
        
        capacity = (C*inputs['stored-ion']['charge']*ct.faraday
                *inputs['eps_solid'])*inputs['thickness']/3600
        bulk_obj.X = X_o
            
        # State variables: electrode potential, electrolyte potential, 
        # electrode composition (nsp), electrolyte composition (nsp)
        nVars = 2 + bulk_obj.n_species + elyte_obj.n_species

        # Load the residual model and store it as a method of this class:
        from .functions import residual, make_alg_consistent


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

    # Save the indices of any algebraic variables:
    if electrode_name=='anode':
        electrode.algvars = [offset + electrode.SVptr['phi_ed']]
    elif electrode_name=='cathode':
        electrode.algvars = [offset + electrode.SVptr['phi_ed']]#[]

    # Load intial state variables:
    SV[electrode.SVptr['phi_ed']] = inputs['phi_0']
    SV[electrode.SVptr['phi_dl']] = phi_elyte_0 - inputs['phi_0']
    SV[electrode.SVptr['C_k_ed']] = electrode.bulk_obj.concentrations
    SV[electrode.SVptr['C_k_elyte']] = electrode.elyte_obj.concentrations

    return SV, electrode
