"""
    porous_separator.py

    Class file for porous separator methods
"""

import cantera as ct
import numpy as np

def residual(SV, SVdot, an, self, ca, params):
    """
    Define the resdisual for the state of the porous separator.

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
    N_k_elyte, i_io = electrode_boundary_flux(SV, an, self)
    
    # The ionic current must equal the external current.
    resid[self.SVptr['phi'][0]] = i_io - params['i_ext']

    # Repeat this for the electric potential in the other separator nodes:
    for j in np.arange(self.n_points-1):
        # Calculate the electrolyte species fluxes and the corresponding ionic 
        # current at the boundary between this separator node and the next one 
        # toward the cathode:
        N_k_elyte, i_io = elyte_flux(SV_loc, self, j)

        # The ionic current must equal the external current.
        resid[self.SVptr['phi'][j+1]] = i_io - params['i_ext']

    return resid

def elyte_flux(SV, self, j):
    """
    Calculate the species fluxes and ionic current between two adjacent nodes in the separator.

    This will eventually serve mainly as a pass-thru to an aliased transport function.
    """

    # TEMPORARY: set species fluxes to zero.
    N_k_elyte = np.zeros_like(self.elyte_obj.X)

    # Read out local and adjacent electric potentials:
    phi_loc = SV[self.SVptr['phi'][j]]
    phi_next = SV[self.SVptr['phi'][j+1]]

    # TEMPORARY: ionic current is due entirely to migration (i.e. electric 
    # potential gradient)
    i_io = ((phi_loc - phi_next)*self.sigma_io*self.dyInv
        *self.elyte_microstructure)

    return N_k_elyte, i_io

def electrode_boundary_flux(SV, ed, sep):
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
    
    # Effective distance between node centers. 'elyte_microstructure' is a 
    # multiplying factor on transport parameters (e.g. porosity divided by tau 
    # factor), which here is weighted by thickness
    dy_elyte_eff = 0.5*(sep.dy/sep.elyte_microstructure 
            + ed.dy/ed.elyte_microstructure)

    # Ionic current. The 'i_ext_flag' is -1 for the anode, +1 for rthe 
    # cathode:            
    i_io = ed.i_ext_flag*(phi_elyte_sep 
        - phi_elyte_ed)*sep.sigma_io/dy_elyte_eff

    # Convert this to flux of the lithium ion, using Faraday's Law:
    N_k_elyte[ed.index_Li] = i_io/ct.faraday/ed.elyte_obj.charges[ed.index_Li]

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

def make_alg_consistent(SV, an, sep, ca, params):
    """ 
    All algebraic variables need to begin the simulation in a 'consistent' state. This means that, when the simulation begins and boudary conditions are applied, the current state of the solution vector must satisfy all algebraic constraints.  In the separator, this means that the electric potential gradient must be consistent with the applied external current.
    """

    # Get the distance between separator node center and anode node center at 
    # separator boundary, plus the electrolyte electric potential in the anode 
    # at the separator boundary:
    dy_elyte_an, phi_elyte_an = electrode_boundary_potential(SV, an, sep)
 
    # Set the electric potential in the separator at the anode boundary equal 
    # to that which matches i_ext.
    SV[sep.SVptr['residual'][sep.SVptr['phi'][0]]] = (phi_elyte_an 
        - params['i_ext']*dy_elyte_an/sep.sigma_io)

    # Similarly, for each additional node in the separator, set the electric 
    # potential such that the ionic current equals i_ext:
    for j in np.arange(sep.n_points-1):
        SV[sep.SVptr['residual'][sep.SVptr['phi'][j+1]]] = \
            (SV[sep.SVptr['residual'][sep.SVptr['phi'][j] ]]
            - params['i_ext']*sep.dy/sep.sigma_io/sep.elyte_microstructure)

    # Get the distance between separator node center and cathode node center at 
    # separator boundary, plus the electrolyte electric potential in the 
    # cathode at the separator boundary:
    dy_elyte_ca, phi_elyte_ca = electrode_boundary_potential(SV, ca, sep)

    # Set the electrolyte electric potential in the cathode at the separator 
    # boundary equal to that which matches i_ext.
    phi_elyte_ca = (SV[sep.SVptr['residual'][sep.SVptr['phi'][-1]]]
        - params['i_ext']*dy_elyte_ca/sep.sigma_io)
    phi_ca = SV[ca.SVptr['residual'][ca.SVptr['phi_ed']]]

    # This is stored as the double layer potential:
    SV[ca.SVptr['residual'][ca.SVptr['phi_dl']]] = phi_elyte_ca - phi_ca

    # Return the updated solution vector:
    return SV
