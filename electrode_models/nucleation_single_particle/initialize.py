"""
    single_particle_electrode.py

    Class file for single-particle (SPM) electrode methods
"""

import cantera as ct
import numpy as np

def initialize(input_file, inputs, sep_inputs, counter_inputs, electrode_name, 
        params, offset):
    """
    Initialize the model.
    """ 
    class electrode:
        """
        Create an electrode object representing the dense electrode
        """

        # Import relevant Cantera objects.
        gas_obj = ct.Solution(input_file, inputs['gas-phase'])
        elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])
        air_elyte_obj = ct.Interface(input_file, inputs['elyte-iphase'], [gas_obj, elyte_obj])
        bulk_obj = ct.Solution(input_file, inputs['bulk-phase'])
        cat_obj = ct.Solution(input_file, inputs['catalyst-phase'])
        product_obj = ct.Solution(input_file, inputs['product-phase'])
        surf_obj = ct.Interface(input_file, inputs['surf-phase'], 
            [bulk_obj, elyte_obj])
        cat_surf_obj = ct.Interface(input_file, inputs['surf-phase'], 
            [cat_obj, elyte_obj])

        # Anode or cathode? Positive external current delivers positive charge 
        # to the anode, and removes positive charge from the cathode.
        name = electrode_name
        if name=='anode':
            i_ext_flag = -1
        elif name=='cathode':
            i_ext_flag = 1
        else:
            raise ValueError("Electrode must be an anode or a cathode.")

        # Store the species index of the Li ion in the Cantera object for the 
        # electrolyte phase:
        #index_Li = elyte_obj.species_index(inputs['mobile-ion'])
        
        # Electrode thickness and inverse thickness:
        dy = inputs['thickness']
        dyInv = 1/dy
        buckets = inputs['buckets']

        # Phase volume fractions
        eps_solid = inputs['eps_solid']
        eps_elyte = 1 - eps_solid

        # Electrode-electrolyte interface area, per unit geometric area.
        # This calculation assumes spherical particles of a single radius, with 
        # no overlap.
        A_surf_ratio = (3*eps_solid*dy/inputs['r_p'])

        # Inverse double layer capacitance, per unit interfacial area.
        C_dl_Inv = 1/inputs['C_dl']

        # Microstructure-based transport scaling factor, based on Bruggeman 
        # coefficient of -0.5:
        elyte_microstructure = eps_elyte**1.5
        
        # SV_offset specifies the index of the first SV variable for the 
        # electrode (zero for anode, nVars_anode + nVars_sep for the cathode)
        SV_offset = offset

        # Determine the electrode capacity (Ah/m2)

        capacity = (inputs['stored-species']['charge']*ct.faraday
                 *inputs['eps_elyte'])*inputs['thickness']/(3600
                 *inputs['stored-species']['MW'])  #Lu (2013) Units?

        # # Return Cantera object composition to original value:
        # bulk_obj.X = X_o
        
        # Number of state variables: electrode potential, electrolyte 
        # potential, electrode composition (n_species), electrolyte composition 
        # (n_species)
        radius = np.linspace(inputs['Minrad'], inputs['Maxrad'], buckets)
        nVars = 3 + elyte_obj.n_species + buckets

        # Load the residual function and other required functions and store 
        # them as methods of this class:
        from .functions import residual, voltage_lim

    # Set Cantera object state:
    electrode.bulk_obj.electric_potential = inputs['phi_0']
    # If the user provided an initial composition, use that, here:
    if 'X_0' in inputs:
        electrode.bulk_obj.TPX = (params['T'], params['P'], inputs['X_0'])
    else:
        electrode.bulk_obj.TP = params['T'], params['P']

    electrode.elyte_obj.TP = params['T'], params['P']
    electrode.surf_obj.TP = params['T'], params['P']
    electrode.conductor_obj.TP = params['T'], params['P']

    # Initialize the solution vector for the electrode domain:
    SV = np.zeros([electrode.nVars])

    # Set up pointers to specific variables in the solution vector:
    electrode.SVptr = {}
    electrode.SVptr['phi_ed'] = np.array([0])
    electrode.SVptr['phi_dl'] = np.array([1])
    electrode.SVptr['Area'] = np.arange([3])
    electrode.SVptr['C_k_elyte'] = np.arange(3, 3 + electrode.elyte_obj.n_species)
    electrode.SVptr['Histogram'] =  (np.arange(3 + electrode.elyte_obj.n_species, 3 +  
        electrode.elyte_obj.n_species + electrode.buckets))

    # A pointer to where the SV varaibles for this electrode are, within the 
    # overall solution vector for the entire problem:
    electrode.SVptr['residual'] = np.arange(offset, offset+electrode.nVars)


    # Save the indices of any algebraic variables:
    electrode.algvars = offset + electrode.SVptr['phi_ed'][:]


    # Load intial state variables:
    SV[electrode.SVptr['phi_ed']] = inputs['phi_0']
    SV[electrode.SVptr['phi_dl']] = sep_inputs['phi_0'] - inputs['phi_0']
    SV[electrode.SVptr['Area']] = inputs['A_0']
    SV[electrode.SVptr['C_k_elyte']] = electrode.elyte_obj.concentrations
    SV[electrode.SVptr['Histogram']] = np.zeros_like(electrode.radius)

    return SV, electrode

"""Citations"""
#Lu, Y.; et al, Energy Environ. Sci., 2013, vol. 6, 750-768, DOI: 10.1039/C3EE23966G
#Official Soundtrack: 
   # Radwimps - Your Name.
   # Walk the Moon - What If Nothing
   # Far East Movement - Identity    