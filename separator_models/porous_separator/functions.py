"""
    porous_separator.py

    Class file for porous separator methods
"""

import cantera as ct
import numpy as np

def residual(SV, SVdot, an, self, ca, params):
    """
    Define the residual for the state of the porous separator.

    This is an array of differential and algebraic governing equations, one for each state variable in the separator.

    1. The electric potentials are algebraic variables, and must be such that the ionic current is spatially invariant (i.e. it is constant and equal to the external applied current, for galvanostatic simulations).  

    The residual corresponding to these variables (suppose an index 'j') are of the form:
        resid[j]  = (epression equaling zero; in this case i_io[j] - i_ext)

    2. All other variables are governed by differential equations, i.e. we have a means to calculate dSV[j]/dt for a state variable SV[j] (state variable with index j).  The residuals corresponding to these variables will have the form:
        resid[j] = SVdot[j] - (expression equalling dSV/dt)
    """

    # Save local copies of the solution vectors, pointers for the separator:
    SVptr = self.SVptr
    SV_loc = SV[SVptr['residual']]
    SVdot_loc = SVdot[SVptr['residual']]

    # Initialize the residual vector, assuming dSVdt = 0 (we will overwrite/
    #  replace this, as necessary)
    resid = SVdot[self.SVptr['residual']]#np.zeros((self.nVars,))
    
    # Calculate the electrolyte species fluxes and the corresponding ionic 
    # current at the anode boundary:
    N_k_elyte, i_io = electrode_boundary_flux(SV, an, self, params['T'])
    # The ionic current must equal the external current.
    resid[self.SVptr['phi'][0]] = i_io - params['i_ext']
    
    # Repeat this for the electric potential in the other separator nodes:
    for j in np.arange(self.n_points-1):
        N_k_elyte_in = N_k_elyte
        # Calculate the electrolyte species fluxes and the corresponding ionic 
        # current at the boundary between this separator node and the next one 
        # toward the cathode:
        N_k_elyte, i_io = elyte_flux(SV_loc, self, j, params['T'])
        
        # The ionic current must equal the external current.
        resid[self.SVptr['phi'][j+1]] = i_io - params['i_ext']
        
        resid[self.SVptr['C_k_elyte'][j]] = \
            (SVdot_loc[self.SVptr['C_k_elyte'][j]] - (N_k_elyte_in - N_k_elyte)
            * self.dyInv / self.eps_elyte)

    j = self.n_points-1
    N_k_elyte_in = N_k_elyte    
    N_k_elyte, i_io = electrode_boundary_flux(SV, ca, self, params['T'])

    resid[self.SVptr['C_k_elyte'][j]] = (SVdot_loc[self.SVptr['C_k_elyte'][j]] 
        - (N_k_elyte_in - N_k_elyte) * self.dyInv / self.eps_elyte) 

    return resid

def elyte_flux(SV, self, j, T):
    """
    Calculate the species fluxes and ionic current between two adjacent nodes in the separator.

    This will eventually serve mainly as a pass-thru to an aliased transport function.
    """

    # TEMPORARY: set species fluxes to zero.
    N_k_elyte = np.zeros_like(self.elyte_obj.X)

    # Read out local and adjacent electric potentials:
    phi_loc = SV[self.SVptr['phi'][j]]
    phi_next = SV[self.SVptr['phi'][j+1]]

    D_k = np.ones_like(self.elyte_obj.X)*20E-12
    C_k_loc = SV[self.SVptr['C_k_elyte'][j]]
    C_k_next = SV[self.SVptr['C_k_elyte'][j+1]]
    C_k_int = 0.5*(C_k_loc + C_k_next)
    D_k_mig = D_k*self.elyte_obj.charges*ct.faraday*C_k_int/ct.gas_constant/T
    
    # Dilute solution theory:
    N_k_elyte = ((D_k*(C_k_loc - C_k_next) + D_k_mig*(phi_loc - phi_next))
        *self.dyInv*self.elyte_microstructure)
    
    # Ionic current = sum(z_k*N_k*F)
    i_io = ct.faraday*np.dot(N_k_elyte, self.elyte_obj.charges)

    return N_k_elyte, i_io

def electrode_boundary_flux(SV, ed, sep, T):
    """
    Calculate the species fluxes and ionic current between a node in the separator and one of the electrodes.

    This will eventually serve mainly as a pass-thru to an aliased transport function.
    """

    # Determine which indices are at the electrode/electrolyte boundary:
    if ed.name=='anode':
        j_ed = -1
        j_elyte = 0
    elif ed.name=='cathode':
        j_ed = 0
        j_elyte = -1

    # Initialize the species fluxes:
    N_k_elyte = np.zeros_like(ed.elyte_obj.X)

    # Elyte electric potential in the electrode:
    phi_ed = SV[ed.SVptr['residual'][ed.SVptr['phi_ed'][j_ed]]]
    phi_dl = SV[ed.SVptr['residual'][ed.SVptr['phi_dl'][j_ed]]]
    phi_elyte_ed = phi_ed + phi_dl
    
    # Elyte electric potential in separator:
    phi_elyte_sep = SV[sep.SVptr['residual'][sep.SVptr['phi'][j_elyte]]]
    
    # Distance between node centers. 
    dy_elyte = 0.5*(sep.dy + ed.dy)
    # 'elyte_microstructure' is a multiplying factor on transport parameters 
    # (e.g. porosity divided by tau factor), which we weight by thickness, here:
    microstructure = ((sep.dy*sep.elyte_microstructure 
        + ed.dy*ed.elyte_microstructure)/dy_elyte/2)

    # TEMPORARY:
    D_k = np.ones_like(sep.elyte_obj.X)*20E-12
    C_k_sep = SV[sep.SVptr['residual'][sep.SVptr['C_k_elyte'][j_elyte]]]
    C_k_ed = SV[ed.SVptr['residual'][ed.SVptr['C_k_elyte'][j_ed]]]
    C_k_int = (C_k_sep*ed.dy + C_k_ed*sep.dy) / (ed.dy + sep.dy)
    D_k_mig = D_k*sep.elyte_obj.charges*ct.faraday*C_k_int/ct.gas_constant/T
    #print(ed.name, C_k_sep, C_k_ed)
    # Dilute solution theory:
    N_k_elyte = ed.i_ext_flag*((D_k * (C_k_sep - C_k_ed) 
        + D_k_mig * (phi_elyte_sep - phi_elyte_ed))
        / dy_elyte) * microstructure
    
    # Ionic current = sum(z_k*N_k*F)
    i_io = ct.faraday*np.dot(N_k_elyte, sep.elyte_obj.charges)

    return N_k_elyte, i_io


def electrode_boundary_potential(SV, ed, sep):
    """
    Calculate the effective distance between node centers at the electrode/electrolyte boundary and the electric potential in the electrolyte phase on the electrode side of this boundary.
    """
    # Elyte electric potential in anode:
    phi_ed = SV[ed.SVptr['residual'][ed.SVptr['phi_ed']]]
    phi_dl = SV[ed.SVptr['residual'][ed.SVptr['phi_dl']]]
    phi_elyte_ed = phi_ed + phi_dl
    
    # Effective distance between node centers, weighted by the electrolyte 
    # microstructure factors:
    dy_elyte_eff = 0.5*(sep.dy/sep.elyte_microstructure 
            + ed.dy/ed.elyte_microstructure)

    return dy_elyte_eff, phi_elyte_ed
