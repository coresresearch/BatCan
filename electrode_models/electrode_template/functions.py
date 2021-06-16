"""
This file provides a template of necessary functions for any new electrode object you might want to create.  Additional methods can, of course, be defined as necessary.

It is suggested that you copy and paste the entire contents of the 'electrode_template' folder, give the pasted folder a relevant name, and then modify as necessary.
"""
def residual(SV, SVdot, self, sep, counter, params):
    """
    Define the residual for the state of the electrode.

    This is an array of differential and algebraic governing equations, one for each state variable in the separator.

    1. The residuals corresponding to algebraic variables (supposing the variable has an index 'j') are of the form:
        resid[j]  = (epression equaling zero)

    2. All other variables are governed by differential equations, i.e. we have a means to calculate dSV[j]/dt for a state variable SV[j] (state variable with index j).  The residuals corresponding to these variables will have the form:
        resid[j] = SVdot[j] - (expression equalling dSV/dt)
    """
    import numpy as np
    import cantera as ct
    
    resid = np.zeros((self.nVars,))

    # Save local copies of the solution vectors, pointers for this electrode:
    SVptr = self.SVptr
    SV_loc = SV[SVptr['residual']]
    SVdot_loc = SVdot[SVptr['residual']]

    # Calculate the residual and return it:

    return resid

def make_alg_consistent(SV, an, sep, ca, params):
    """ 
    All algebraic variables need to begin the simulation in a 'consistent' state. This means that, when the simulation begins and boudary conditions are applied, the current state of the solution vector must satisfy all algebraic constraints.
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