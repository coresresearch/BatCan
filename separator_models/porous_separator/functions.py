"""
    porous_separator.py

    Class file for porous separator methods
"""

import cantera as ct
import numpy as np

def residual(SV, SVdot, an, self, ca, params):

    # Save local copies of the solution vectors, pointers for this electrode:
    SVptr = self.SVptr
    SV_loc = SV[SVptr['residual']]
    SVdot_loc = SVdot[SVptr['residual']]

    resid = SVdot[self.SVptr['residual']]#np.zeros((self.nVars,))
    
    N_k_io = electrode_boundary_flux(SV, an, self)
    i_io = np.dot(N_k_io, self.elyte_obj.charges)*ct.faraday
    
    resid[self.SVptr['phi'][0]] = i_io - params['i_ext']
    for j in np.arange(self.n_points-1):
        N_k_elyte, i_io = elyte_flux(SV_loc, self, j)
        resid[self.SVptr['phi'][j+1]] = i_io - params['i_ext']

    return resid

def elyte_flux(SV, self, j):
    # TEMPORARY:
    N_k_elyte = np.zeros_like(self.elyte_obj.X)

    phi_loc = SV[self.SVptr['phi'][j]]
    phi_next = SV[self.SVptr['phi'][j+1]]
    i_io = (phi_loc - phi_next)*self.sigma_io*self.dyInv

    return N_k_elyte, i_io

def electrode_boundary_flux(SV, ed, sep):
    if ed.name=='anode':
        j_ed = -1
        j_elyte = 0
    elif ed.name=='cathode':
        j_ed = 0
        j_elyte = -1

    N_k_elyte = np.zeros_like(ed.elyte_obj.X)

    # Elyte electric potential in cathode:
    phi_ed = SV[ed.SVptr['residual'][ed.SVptr['phi_ed'][j_ed]]]
    phi_dl = SV[ed.SVptr['residual'][ed.SVptr['phi_dl'][j_ed]]]
    phi_elyte_ed = phi_ed + phi_dl
    
    # Elyte electric potential in separator:
    phi_elyte_sep = SV[sep.SVptr['residual'][sep.SVptr['phi'][j_elyte]]]
    
    # Ionic current:
    dy_elyte_avg = 0.5*(sep.dy/sep.elyte_microstructure 
            + ed.dy/ed.elyte_microstructure)
    i_io = ed.i_ext_flag*(phi_elyte_sep 
        - phi_elyte_ed)*sep.sigma_io/dy_elyte_avg

    # Convert this to flux of the lithium ion:
    N_k_elyte[ed.index_Li] = i_io/ct.faraday/ed.elyte_obj.charges[ed.index_Li]

    return N_k_elyte


def electrode_boundary_potential(SV, ed, sep):
    # Elyte electric potential in anode:
    phi_ed = SV[ed.SVptr['residual'][ed.SVptr['phi_ed']]]
    phi_dl = SV[ed.SVptr['residual'][ed.SVptr['phi_dl']]]
    phi_elyte_ed = phi_ed + phi_dl
    
    # Ionic resistance:
    dy_elyte_avg = 0.5*(sep.dy/sep.elyte_microstructure 
            + ed.dy/ed.elyte_microstructure)

    return dy_elyte_avg, phi_elyte_ed

def make_alg_consistent(SV, an, sep, ca, params):
    """ All algebraic variables need to begin the simulation in a 'consistent' state. This means that, when the simulation begins and boudary conditions are applied, the current state of the solution vector must satisfy all algebraic constraints.  In the separator, this means that the electric potential gradient must be consistent with the applied external current."""

    dy_elyte_an, phi_elyte_an = electrode_boundary_potential(SV, an, sep)

    SV[sep.SVptr['residual'][sep.SVptr['phi'][0]]] = (phi_elyte_an 
        - params['i_ext']*dy_elyte_an/sep.sigma_io)

    for j in np.arange(sep.n_points-1):
        SV[sep.SVptr['residual'][sep.SVptr['phi'][j+1]]] = \
            (SV[sep.SVptr['residual'][sep.SVptr['phi'][j] ]]
            - params['i_ext']*sep.dy/sep.sigma_io)

    # We are going to cheat and use the anode function for the cathode :)
    dy_elyte_ca, phi_elyte_ca = electrode_boundary_potential(SV, ca, sep)
    phi_elyte_ca = (SV[sep.SVptr['residual'][sep.SVptr['phi'][-1]]]
        - params['i_ext']*dy_elyte_ca/sep.sigma_io)

    # SV[sep.SVptr['residual'][sep.SVptr['phi']]] = phi_elyte_sep
    phi_ca = SV[ca.SVptr['residual'][ca.SVptr['phi_ed']]]
    SV[ca.SVptr['residual'][ca.SVptr['phi_dl']]] = phi_elyte_ca - phi_ca

    return SV
