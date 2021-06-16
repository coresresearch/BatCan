"""
    electrode_template.py

    Class file for a newly-implemented electrode type.
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
        
        # Number of state variables: electrode potential, electrolyte 
        # potential, electrode composition (nsp), electrolyte composition (nsp)
        nVars = 0

        # Load the residual model and store it as a method of this class:
        from .functions import residual, make_alg_consistent, voltage_lim

    # Set Cantera object state:
    electrode.bulk_obj.electric_potential = inputs['phi_0']
    # If the user provided an initial composition, use that here:
    if 'X_0' in inputs:
        electrode.bulk_obj.TPX = (params['T'], params['P'], inputs['X_0'])
    else:
        electrode.bulk_obj.TP = params['T'], params['P']

    electrode.elyte_obj.TP = params['T'], params['P']
    electrode.surf_obj.TP = params['T'], params['P']
    electrode.conductor_obj.TP = params['T'], params['P']

    SV = np.zeros([electrode.nVars])

    # Set up pointers to variable locations in the solution vector:
    electrode.SVptr = {}
    # For example:
    electrode.SVptr['phi_ed'] = 0 

    # Save location of this element's varaibles in the larger solution vector:
    electrode.SVptr['residual'] = np.arange(offset, offset+electrode.nVars)

    # Save the indices of any algebraic variables:
    electrode.algvars = [offset + electrode.SVptr['phi_ed']]

    # Load intial state variables:
    # For example:
    SV[electrode.SVptr['phi_ed']] = inputs['phi_0']

    return SV, electrode
