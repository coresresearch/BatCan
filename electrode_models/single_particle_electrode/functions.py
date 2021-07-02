def residual(SV, SVdot, electrode, sep, counter, params):
    """
    Define the residual for the state of the single particle electrode.

    This is an array of differential and algebraic governing equations, one for each state variable in the anode (anode plus a thin layer of electrolyte + separator).

    1. The electric potential in the electrode phase is an algebraic variable.
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
    
    # Initialize the residual:
    resid = np.zeros((electrode.nVars,))

    # Save local copies of the solution vectors, pointers for this electrode:
    SVptr = electrode.SVptr
    SV_loc = SV[SVptr['electrode']]
    SVdot_loc = SVdot[SVptr['electrode']]

    # Read the electrode and electrolyte electric potential:
    phi_ed = SV_loc[SVptr['phi_ed']]
    phi_elyte = phi_ed + SV_loc[SVptr['phi_dl']]

    # Read out electrode bulk composition; set the Cantra object:
    C_k_ed = SV_loc[SVptr['C_k_ed']] # Molar density (kmol/m3 of phase)
    X_k_ed = C_k_ed/sum(C_k_ed) # Mole fraction
    electrode.bulk_obj.X = X_k_ed

    # Set electric potentials for Cantera objects:
    electrode.bulk_obj.electric_potential = phi_ed
    electrode.conductor_obj.electric_potential = phi_ed
    electrode.elyte_obj.electric_potential = phi_elyte
    
    # Faradaic current density is positive when electrons are consumed 
    # (Li transferred to the electrode)
    i_Far = -(ct.faraday 
        * electrode.surf_obj.get_net_production_rates(electrode.conductor_obj))
    
    # Double layer current has the same sign as i_Far:
    i_dl = electrode.i_ext_flag*params['i_ext']/electrode.A_surf_ratio - i_Far
    N_k_sep, i_io = sep.electrode_boundary_flux(SV, electrode, sep, params['T'])

    if electrode.name=='anode':
        # The electric potential of the anode = 0 V.
        resid[[SVptr['phi_ed'][0]]] = SV_loc[SVptr['phi_ed'][0]]
    elif electrode.name=='cathode':
        # For the cathode, the potential of the cathode must be such that the 
        # electrolyte electric potential (calculated as phi_ca + dphi_dl) 
        # produces the correct ionic current between the separator and cathode:
        resid[SVptr['phi_ed']] = i_io - params['i_ext']

    # Differential equation for the double layer potential:
    resid[SVptr['phi_dl']] = \
        SVdot_loc[SVptr['phi_dl']] - i_dl*electrode.C_dl_Inv

    # TEMPORARY: Set time derivatives for the species concentrations to zero:
    resid[SVptr['C_k_ed']] = SVdot_loc[SVptr['C_k_ed']] 

    # Molar production rate of electrode species (kmol/m2/s).
    sdot_elyte = \
        electrode.surf_obj.get_net_production_rates(electrode.elyte_obj)
    
    # Double layer current removes Li from the electrolyte.  Subtract this from 
    # sdot_electrolyte:
    sdot_elyte[electrode.index_Li] -= i_dl / ct.faraday
        
    # Change in electrolyte species concentration per unit time:
    dCk_elyte_dt = \
        ((sdot_elyte * electrode.A_surf_ratio + electrode.i_ext_flag * N_k_sep)* electrode.dyInv / electrode.eps_elyte)
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
    Sometimes, an electrode object requires adjustments to the separator object.  This is not the case, for the SPM.
    """

    # Return the separator class object, unaltered:
    return sep