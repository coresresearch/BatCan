"""
    ionic_resistor.py

    Class file for ionic resistor separator methods. This is a very simple separator model, that considers it to have an ionic resistance, but no chemical variation.
"""

import cantera as ct
import numpy as np

def residual(SV, SVdot, an, self, ca, params):
    """
    Define the residual for the state of the separator.

    This is a single algebraic governing equation to determine the separator electric potential.  The separator electric potential must be such that the ionic current is spatially invariant (i.e. it is constant and equal to the external applied current, for galvanostatic simulations).  

    The residual corresponding to this variable (suppose an index 'j') is of the form:
            resid[j]  = (epression equaling zero; here i_io - i_ext)
    """
    # Initialize the residual vector, assuming dSVdt = 0 (we will overwrite/
    #  replace this, as necessary)
    resid = SVdot[self.SVptr['residual']]

    # Calculate the distance to the anode node center and the anode's electrolyte phase electric potential at the separator boundary:
    dy, phi_elyte_an = electrode_boundary_potential(SV, an, self)

    # Calculate the electric potential that satisfies the algebraic equation:
    phi_elyte_sep = phi_elyte_an - params['i_ext']*dy/self.sigma_io
    
    # Calculate the residual:
    resid[self.SVptr['phi']] = (SV[self.SVptr['residual'][self.SVptr['phi']]] 
            - phi_elyte_sep)

    return resid

def electrode_boundary_flux(SV, ed, sep):
    """
    Calculate the species fluxes and ionic current between a node in the separator and one of the electrodes.
    """
    N_k_elyte = np.zeros_like(ed.elyte_obj.X)

    # Elyte electric potential in cathode:
    phi_ed = SV[ed.SVptr['residual'][ed.SVptr['phi_ed']]]
    phi_dl = SV[ed.SVptr['residual'][ed.SVptr['phi_dl']]]
    phi_elyte_ca = phi_ca + phi_dl
    
    # Elyte electric potential in separator:
    phi_elyte_sep = SV[sep.SVptr['residual'][sep.SVptr['phi']]]
    
    # Average electronic resistance:
    R_io_avg = 0.5*(sep.dy/sep.elyte_microstructure 
            + ed.dy/ed.elyte_microstructure)

    # Ionic current:
    i_io = (phi_elyte_sep - phi_elyte_ca)*sep.sigma_io/R_io_avg

    # Convert this to flux of the lithium ion:
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
    dy_elyte, phi_elyte_an = electrode_boundary_potential(SV, an, sep)

    # Calculate the separator electric potential which satisfied the algebraic 
    # constraint:
    phi_elyte_sep = phi_elyte_an - params['i_ext']*dy_elyte/sep.sigma_io

    # Modify SV to satisfy the constraint:
    SV[sep.SVptr['residual'][sep.SVptr['phi']]] = phi_elyte_sep

    # Get the distance between separator node center and cathode node center at 
    # separator boundary, plus the electrolyte electric potential in the 
    # cathode at the separator boundary:
    dy_elyte, phi_elyte_ca = electrode_boundary_potential(SV, ca, sep)
    # Set the electrolyte electric potential in the cathode at the separator 
    # boundary equal to that which matches i_ext.
    phi_elyte_ca = phi_elyte_sep - params['i_ext']*dy_elyte/sep.sigma_io
    phi_ca = SV[ca.SVptr['residual'][ca.SVptr['phi_ed']]]

    # Modify SV to satisfy the constraint. This is stored as the double layer 
    # potential:
    SV[ca.SVptr['residual'][ca.SVptr['phi_dl']]] = phi_elyte_ca - phi_ca

    # Return the updated solution vector:
    return SV