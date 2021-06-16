def residual(SV, SVdot, self, sep, counter, params):
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
    """
    import numpy as np
    import cantera as ct
    
    # Initialize the residual:
    resid = np.zeros((self.nVars,))

    # Save local copies of the solution vectors, pointers for this electrode:
    SVptr = self.SVptr
    SV_loc = SV[SVptr['residual']]
    SVdot_loc = SVdot[SVptr['residual']]

    # Read the electrode and electrolyte electric potential:
    phi_ed = SV_loc[SVptr['phi_ed']]
    phi_elyte = phi_ed + SV_loc[SVptr['phi_dl']]

    # Read out electrode bulk composition; set the Cantra object:
    C_k_ed = SV_loc[SVptr['C_k_ed']] # Molar density (kmol/m3 of phase)
    X_k_ed = C_k_ed/sum(C_k_ed) # Mole fraction
    self.bulk_obj.X = X_k_ed

    # Set electric potentials for Cantera objects:
    self.bulk_obj.electric_potential = phi_ed
    self.conductor_obj.electric_potential = phi_ed
    self.elyte_obj.electric_potential = phi_elyte
    
    # Faradaic current density is positive when electrons are consumed 
    # (Li transferred to the anode)
    sdot_electron = self.surf_obj.get_net_production_rates(self.conductor_obj)
    i_Far = -ct.faraday*sdot_electron
    
    # Double layer current has the same sign as i_Far:
    i_dl = self.i_ext_flag*params['i_ext']/self.A_surf_ratio - i_Far
    
    if self.name=='anode':
        # The electric potential of the anode = 0 V.
        resid[SVptr['residual'][SVptr['phi_ed']]] = SV_loc[SVptr['phi_ed']]
    elif self.name=='cathode':
        # For the cathode, the potential of the cathode must be such that the 
        # electrolyte electric potential (calculated as phi_ca + dphi_dl) 
        # produces the correct ionic current between the separator and cathode:
        N_k_sep, i_io = sep.electrode_boundary_flux(SV, self, sep)
                        
        resid[SVptr['phi_ed']] = i_io - params['i_ext']

    # Differential equation for the double layer potential:
    resid[SVptr['phi_dl']] = (SVdot_loc[SVptr['phi_dl']] - i_dl*self.C_dl_Inv)

    # TEMPORARY: Set time derivatives for the species concentrations to zero:
    resid[SVptr['C_k_ed']] = SVdot_loc[SVptr['C_k_ed']] 
    # -         sdot_bulk_obj*A_surf_ratio*self.eps_bulk)
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
    Sometimes, an electrode object requires adjustments to the separator object.  This is not the case, for the SPM.
    """

    # Return the separator class object, unaltered:
    return sep