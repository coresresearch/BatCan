# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:37:06 2018

@author: dkorff
"""

"""====================================================================="""
"""===========================Model Functions==========================="""
"""====================================================================="""
import cantera as ct
import numpy as np
from li_ion_battery_p2d_inputs import Inputs

def set_state(offset, SV, ed, surf, el, conductor, ptr):
    
    if ed.name == Inputs.cathode_phase:              
        ed.X = [SV[offset + ptr['X_ed'][-1]], 1 - SV[offset + ptr['X_ed'][-1]]]
    
    ed.electric_potential = SV[offset + ptr['Phi_ed']]
    
    conductor.electric_potential = SV[offset + ptr['Phi_ed']]
    
    el.X = SV[offset + ptr['X_k_elyte']]
    el.electric_potential = SV[offset + ptr['Phi_dl']] + SV[offset + ptr['Phi_ed']]
    
    state = {}
    state['sdot'] = surf.get_net_production_rates(el)
    state['phi_ed'] = SV[offset + ptr['Phi_ed']]
    state['phi_el'] = SV[offset + ptr['Phi_dl']] + SV[offset + ptr['Phi_ed']]
    state['rho_ed'] = ed.density_mole
    state['rho_el'] = el.density_mole
    state['X_k_el'] = SV[offset + ptr['X_k_elyte']]
    
    return state

"""====================================================================="""

def set_state_sep(offset, SV, el, ptr):
    
    el.X = SV[offset + ptr['X_k_elyte']]
    el.electric_potential = SV[offset + ptr['Phi']]
    
    state = {}
    state['phi_el'] = SV[offset + ptr['Phi']]
    state['rho_el'] = el.density_mole
    state['X_k_el'] = SV[offset + ptr['X_k_elyte']]
    
    return state

"""====================================================================="""

def dilute_flux(s1, s2, dyInv, ed, D_k, D_migr_k):
    F = ct.faraday; R = ct.gas_constant; T = Inputs.T
    z_k = Inputs.z_k_elyte
    # Calculate ionic flux in electrolyte
    
    # Total molar concentration in electrolyte
    C_0 = (s2['rho_el'] + s1['rho_el'])/2.
    
    N_io_p = (- D_k*C_0*(s2['X_k_el'] - s1['X_k_el'])*dyInv 
              - D_migr_k*(z_k*F/R/T)*(s2['phi_el'] - s1['phi_el'])*dyInv)
    
#    N_io_p = (-ed.u_Li_elyte*(R*T*C_0*(s2['X_k_el'] - s1['X_k_el'])
#    + Inputs.z_k_elyte*F*C_k*(s2['phi_el'] - s1['phi_el']))*dyInv)
    
    i_io_p = np.dot(N_io_p, Inputs.z_k_elyte)*F
    
    return N_io_p, i_io_p

"""====================================================================="""

def solid_flux(SV, offset, ptr, s1, ed):
    
    X_Li = SV[offset + ptr['X_ed']]
    DiffFlux = np.zeros([ed.nshells+1])
    DiffFlux[1:-1] = ed.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/ed.dr
    DiffFlux[-1] = -s1['sdot'][ed.ptr['iFar']]/s1['rho_ed']
    
    return DiffFlux

def setup_plots(plt, rate_tag):
    if Inputs.plot_potential_profiles == 1:
        fig1, axes1 = plt.subplots(sharey="row", figsize=(14,6), nrows=1, 
                                   ncols = 2+(Inputs.flag_re_equil*Inputs.phi_time))
        plt.subplots_adjust(wspace = 0.15, hspace = 0.4)
        fig1.text(0.15, 0.8, rate_tag, fontsize=20, 
                  bbox=dict(facecolor='white', alpha=0.5))
        
    if Inputs.plot_electrode_profiles == 1:
        nrows = Inputs.flag_anode + Inputs.flag_cathode
        ncols = 2 + Inputs.flag_re_equil
        fig2, axes2 = plt.subplots(sharey="row", figsize=(18,9), nrows=nrows,
                                   ncols=ncols)
        plt.subplots_adjust(wspace=0.15, hspace=0.4)
        fig2.text(0.15, 0.8, rate_tag, fontsize=20, 
                  bbox=dict(facecolor='white', alpha=0.5))
        
    if Inputs.plot_elyte_profiles == 1:
        nrows = Inputs.flag_anode + Inputs.flag_cathode + Inputs.flag_sep
        ncols = 2 + Inputs.flag_re_equil
        fig3, axes3 = plt.subplots(sharey="row", figsize=(18,9), nrows=nrows,
                                    ncols=ncols)
        plt.subplots_adjust(wspace=0.15, hspace=0.4)
        fig3.text(0.15, 0.8, rate_tag, fontsize=20, 
                  bbox=dict(facecolor='white', alpha=0.5))
        
    return fig1, axes1, fig2, axes2, fig3, axes3

