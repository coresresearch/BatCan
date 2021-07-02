def residual(SV, SVdot, electrode, sep, counter, params):
    """
    Define the residual for the state of the dense electrode.

    This is an array of differential and algebraic governing equations, one for each state variable in the anode (anode plus a thin layer of electrolyte + separator).

    1. The electric potential is an algebraic variable.
        In the anode, phi = 0 is the reference potential for the system.
        In the cathode, the electric potential must be such that the ionic current is spatially invariant (i.e. it is constant and equal to the external applied current, for galvanostatic simulations).  

        The residual corresponding to these variables (suppose an index 'j') are of the form:
            resid[j]  = (epression equaling zero)

    2. All other variables are governed by differential equations.
    
        We have a means to calculate dSV[j]/dt for a state variable SV[j] (state variable with index j).  
    
        The residuals corresponding to these variables will have the form:
            resid[j] = SVdot[j] - (expression equalling dSV/dt)

    Inputs:
        - SV: the solution vector representing the state of the entire battery domain.
        - SVdot: the time derivative of each state variable: dSV/dt
        - electrode: the object representing the current electrode
        - sep: the object representing the separator
        - counter: the object representing the electrode counter to the current electrode
        - params: dict of battery simulation parameters.
    """
    import numpy as np
    import cantera as ct
    from math import tanh
    
    # Initialize the residual array:
    resid = np.zeros((electrode.nVars,))

    # Save local copies of the solution vectors, pointers for this electrode:
    SVptr = electrode.SVptr
    SV_loc = SV[SVptr['electrode']]
    SVdot_loc = SVdot[SVptr['electrode']]

    # Read electrode and electrolyte electric potentials:
    phi_ed = SV_loc[SVptr['phi_ed']]
    phi_elyte = phi_ed + SV_loc[SVptr['phi_dl']]

    # Set electric potentials for Cantera objects:
    electrode.bulk_obj.electric_potential = phi_ed
    electrode.conductor_obj.electric_potential = phi_ed
    electrode.elyte_obj.electric_potential = phi_elyte
    
    # Multiplier on the electrode removal reaction. Quickly goes to zero, for 
    # thicknesses below a user-specified minimum:
    mult = tanh(SV_loc[SVptr['thickness']]/electrode.min_thickness)

    # Molar production rate of electrons in the 'conductor' phase. Electrons are created when lithium is consumed. We scale the the electron creation rate, then, by our multiplier.  When the anode thickness is below the minimum, the electron creation rate goes quickly to zero, but the electron destruction rate is unaffected:
    sdot_electron = \
        (mult*electrode.surf_obj.get_creation_rates(electrode.conductor_obj)
        - electrode.surf_obj.get_destruction_rates(electrode.conductor_obj))

    # Molar production rate of electrode species (kmol/m2/s). Here, we scale 
    # the destruction rate by our multiplier.
    sdot_electrode = (electrode.surf_obj.get_creation_rates(electrode.bulk_obj)
            - mult*electrode.surf_obj.get_destruction_rates(electrode.bulk_obj))

    # Faradaic current density is positive when electrons are consumed 
    # (Li transferred to the anode)
    i_Far = -ct.faraday*sdot_electron
    
    # Double layer current has the same sign as i_Far:
    i_dl = electrode.i_ext_flag*params['i_ext'] / electrode.A_surf_ratio - i_Far
    
    # Differential equation for the double layer potential difference:
    resid[SVptr['phi_dl']] = (SVdot_loc[SVptr['phi_dl']] 
        - i_dl*electrode.C_dl_Inv)

    # Flux of electrolyte species between the separator and the electrolyte in 
    # the current electrode domain:
    N_k_sep, i_io = sep.electrode_boundary_flux(SV, electrode, sep, params['T'])

    # Electrode electric potential
    if electrode.name=='anode':
        # For the anode, the electric potential is an algebraic variable, 
        # always equal to zero:
        resid[SVptr['phi_ed']] = SV_loc[SVptr['phi_ed']]

    elif electrode.name=='cathode':
        # The electric potential of the electrolyte in the cathode domain must 
        # be such that the ionic current from the separator to the cathode 
        # equals the external current:
                
        resid[SVptr['phi_ed']] = i_io - params['i_ext']
    
    # Change in thickness per time:
    dH_dt = np.dot(sdot_electrode, electrode.bulk_obj.partial_molar_volumes)
    resid[SVptr['thickness']] = SVdot_loc[SVptr['thickness']] - dH_dt
    
    # Set time derivatives for electrolyte species concentrations to zero 
    # (temporary)

    # Molar production rate of electrode species (kmol/m2/s). Here, we scale 
    # the destruction rate by our multiplier.
    sdot_electrolyte = \
        (mult*electrode.surf_obj.get_creation_rates(electrode.elyte_obj)
        - electrode.surf_obj.get_destruction_rates(electrode.elyte_obj))

    # Double layer current removes Li from the electrolyte.  Add this to 
    # sdot_electrolyte:
    sdot_electrolyte[electrode.index_Li] -= i_dl / ct.faraday
    dCk_elyte_dt = \
        (sdot_electrolyte * electrode.A_surf_ratio 
        + electrode.i_ext_flag * N_k_sep) / electrode.dy_elyte 
    resid[SVptr['C_k_elyte']] = SVdot_loc[SVptr['C_k_elyte']] - dCk_elyte_dt

    return resid

def voltage_lim(SV, electrode, val):
    """
    Check to see if the voltage limits have been exceeded.
    """
    # Save local copies of the solution vector and pointers for this electrode:
    SVptr = electrode.SVptr
    SV_loc = SV[SVptr['electrode']]
    
    # Calculate the current voltage, relative to the limit.  The simulation 
    # looks for instances where this value changes sign (i.e. crosses zero)
    voltage_eval = SV_loc[SVptr['phi_ed']] - val

    return voltage_eval

def adjust_separator(electrode, sep):
    """
    The electrode domain considers the electrode object plus a thin layer of the separator, adjacent to the electrode. We subtract this thickness from the total separator thickness, so that we do not inadvertently increase the total transport resistance through the separator.
    """
    sep.dy -= electrode.dy_elyte
    return sep
