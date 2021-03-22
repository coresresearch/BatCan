"""
    porous_separator.py

    Class file for porous separator methods
"""

import cantera as ct
import numpy as np

def residual(SV, SVdot, an, self, ca, params):
    resid = SVdot[self.SVptr['residual']]#np.zeros((self.nVars,))

    R_io, phi_elyte_an = anode_boundary(SV, an, self)

    phi_elyte_sep = phi_elyte_an - params['i_ext']*R_io/self.sigma_io
    
    resid[self.SVptr['phi']] = (SV[self.SVptr['residual'][self.SVptr['phi']]] 
            - phi_elyte_sep)
    return resid

def cathode_boundary(SV, ca, sep):
    N_k_elyte = np.zeros_like(ca.elyte_obj.X)

    # Elyte electric potential in cathode:
    phi_ca = SV[ca.SVptr['residual'][ca.SVptr['phi_ed']]]
    phi_dl = SV[ca.SVptr['residual'][ca.SVptr['phi_dl']]]
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


def anode_boundary(SV, an, sep):
    # Elyte electric potential in anode:
    phi_an = SV[an.SVptr['residual'][an.SVptr['phi_ed']]]
    phi_dl = SV[an.SVptr['residual'][an.SVptr['phi_dl']]]
    phi_elyte_an = phi_an + phi_dl
    
    # Elyte electric potential in separator:
    phi_elyte_sep = SV[sep.SVptr['residual'][sep.SVptr['phi']]]

    # Ionic current:
    R_io_avg = 0.5*(sep.dy/sep.elyte_microstructure 
            + an.dy/an.elyte_microstructure)

    return R_io_avg, phi_elyte_an

def make_alg_consistent(SV, an, sep, ca, params):

    R_io, phi_elyte_an = anode_boundary(SV, an, sep)

    phi_elyte_sep = phi_elyte_an - params['i_ext']*R_io/sep.sigma_io

    # We are going to cheat and use the anode function for the cathode :)
    R_io, phi_elyte_ca = anode_boundary(SV, ca, sep)
    phi_elyte_ca = phi_elyte_sep - params['i_ext']*R_io/sep.sigma_io

    SV[sep.SVptr['residual'][sep.SVptr['phi']]] = phi_elyte_sep
    phi_ca = SV[ca.SVptr['residual'][ca.SVptr['phi_ed']]]
    SV[ca.SVptr['residual'][ca.SVptr['phi_dl']]] = phi_elyte_ca - phi_ca

    return SV