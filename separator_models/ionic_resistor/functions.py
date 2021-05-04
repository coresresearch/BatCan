"""
    porous_separator.py

    Class file for porous separator methods
"""

import cantera as ct
import numpy as np

def residual(SV, SVdot, an, self, ca, params):
    resid = SVdot[self.SVptr['residual']]#np.zeros((self.nVars,))

    dy, phi_elyte_an = electrode_boundary_potential(SV, an, self)

    phi_elyte_sep = phi_elyte_an - params['i_ext']*dy/self.sigma_io
    
    resid[self.SVptr['phi']] = (SV[self.SVptr['residual'][self.SVptr['phi']]] 
            - phi_elyte_sep)
    return resid

def electrode_boundary_flux(SV, ed, sep):
    N_k_elyte = np.zeros_like(ed.elyte_obj.X)

    # Elyte electric potential in cathode:
    phi_ed = SV[ed.SVptr['residual'][ed.SVptr['phi_ed']]]
    phi_dl = SV[ed.SVptr['residual'][ed.SVptr['phi_dl']]]
    phi_elyte_ca = phi_ca + phi_dl
    
    # Elyte electric potential in separator:
    phi_elyte_sep = SV[sep.SVptr['residual'][sep.SVptr['phi']]]
    
    # Ionic current:
    R_io_avg = 0.5*(sep.dy/sep.elyte_microstructure 
            + ca.dy/ca.elyte_microstructure)
    i_io = (phi_elyte_sep - phi_elyte_ca)*sep.sigma_io/R_io_avg

    # Convert this to flux of the lithium ion:
    N_k_elyte[ca.index_Li] = i_io/ct.faraday/ca.elyte_obj.charges[ca.index_Li]

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

    dy_elyte, phi_elyte_an = electrode_boundary_potential(SV, an, sep)

    phi_elyte_sep = phi_elyte_an - params['i_ext']*dy_elyte/sep.sigma_io

    # We are going to cheat and use the anode function for the cathode :)
    dy_elyte, phi_elyte_ca = electrode_boundary_potential(SV, ca, sep)
    phi_elyte_ca = phi_elyte_sep - params['i_ext']*dy_elyte/sep.sigma_io

    SV[sep.SVptr['residual'][sep.SVptr['phi']]] = phi_elyte_sep
    phi_ca = SV[ca.SVptr['residual'][ca.SVptr['phi_ed']]]
    SV[ca.SVptr['residual'][ca.SVptr['phi_dl']]] = phi_elyte_ca - phi_ca

    return SV