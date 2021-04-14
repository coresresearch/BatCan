def residual(SV, SVdot, self, sep, counter, params):
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
    # Routines needed to make the initial solutioon consistent with any 
    # algebraic constraints.
    return SV

def voltage_lim(SV, self, val):
    # Evaluate whether or not the electrode voltage has exceeded any 
    # user-specified bounds.
    SVptr = self.SVptr
    SV_loc = SV[SVptr['residual']]
    
    voltage_eval = SV_loc[SVptr['phi_ed']] - val
    
    return voltage_eval

def add_argument(self, sep):
    return sep