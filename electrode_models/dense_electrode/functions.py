def residual(SV, SVdot, self, sep, counter, params):
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
    """
    import numpy as np
    import cantera as ct
    from math import tanh
    
    # Initialize the residual array:
    resid = np.zeros((self.nVars,))

    # Save local copies of the solution vectors, pointers for this electrode:
    SVptr = self.SVptr
    SV_loc = SV[SVptr['residual']]
    SVdot_loc = SVdot[SVptr['residual']]

    # Read electrode and electrolyte electric potentials:
    phi_ed = SV_loc[SVptr['phi_ed']]
    phi_elyte = phi_ed + SV_loc[SVptr['phi_dl']]

    # Set electric potentials for Cantera objects:
    self.bulk_obj.electric_potential = phi_ed
    self.conductor_obj.electric_potential = phi_ed
    self.elyte_obj.electric_potential = phi_elyte
    
    # Multiplier on the electrode removal reaction. Quickly goes to zero, for 
    # thicknesses below a user-specified minimum:
    mult = tanh(SV_loc[SVptr['thickness']]/self.min_thickness)

    # Molar production rate of electrons in the 'conductor' phase. Electrons are created when lithium is consumed. We scale the the electron creation rate, then, by our multiplier.  When the anode thickness is below the minimum, the electron creation rate goes quickly to zero, but the electron destruction rate is unaffected:
    sdot_electron = (mult*self.surf_obj.get_creation_rates(self.conductor_obj)
            - self.surf_obj.get_destruction_rates(self.conductor_obj))

    # Molar production rate of electrode species (kmol/m2/s). Here, we scale 
    # the destruction rate by our multiplier.
    sdot_electrode = (self.surf_obj.get_creation_rates(self.bulk_obj)
            - mult*self.surf_obj.get_destruction_rates(self.bulk_obj))
    
    # Faradaic current density is positive when electrons are consumed 
    # (Li transferred to the anode)
    i_Far = -ct.faraday*sdot_electron
    
    # Double layer current has the same sign as i_Far:
    i_dl = self.i_ext_flag*params['i_ext']/self.A_surf_ratio - i_Far
    
    if self.name=='anode':
        # For the anode, the electric potential is an algebraic variable, always equal to zero:
        resid[SVptr['residual'][SVptr['phi_ed']]] = SV_loc[SVptr['phi_ed']]

    elif self.name=='cathode':
        # TEMPORARY: phi_elyte in cathode matches that in the anode.
        # TODO #21
        N_k_sep = sep.cathode_boundary(SV, self, sep)
        i_io = np.dot(N_k_sep, self.elyte_obj.charges)*ct.faraday
                
        resid[SVptr['phi_ed']] = i_io - params['i_ext']

    resid[SVptr['phi_dl']] = (SVdot_loc[SVptr['phi_dl']] - i_dl*self.C_dl_Inv)
    
    # Change in thickness per time:
    dH_dt = np.dot(sdot_electrode,self.bulk_obj.partial_molar_volumes)
    resid[SVptr['thickness']] = SVdot_loc[SVptr['thickness']] - dH_dt
    
    # Set time derivatives to zero (temporary)
    resid[SVptr['C_k_elyte']] = SVdot_loc[SVptr['C_k_elyte']]

    return resid

def make_alg_consistent(SV, an, sep, ca, params):
    """ 
    All algebraic variables need to begin the simulation in a 'consistent' state. This means that, when the simulation begins and boudary conditions are applied, the current state of the solution vector must satisfy all algebraic constraints.  
    
    For an anode, there are no constraints. 
    For a cathode, the electric potential gradient must be consistent with the applied external current. However, this is handled by the separator's 'make_alg_consistent' method.  Therefore, nothing needs to be done, here.
    """
    return SV

def voltage_lim(SV, self, val):
    """
    Check to see if the voltage limits have been exceeded.
    """
    # Save local copies of the solution vector and pointers for this electrode:
    SVptr = self.SVptr
    SV_loc = SV[SVptr['residual']]
    
    # Calculate the current voltage, relative to the limit.  The simulation 
    # looks for instances where this value changes sign (i.e. crosses zero)
    voltage_eval = SV_loc[SVptr['phi_ed']] - val

    return voltage_eval

def adjust_separator(self, sep):
    """
    The electrode domain considers the electrode object plus a thin layer of the separator, adjacent to the electrode. We subtract this thickness from the total separator thickness, so that we do not inadvertently increase the total transport resistance through the separator.
    """
    sep.dy -= self.dy_elyte
    return sep
