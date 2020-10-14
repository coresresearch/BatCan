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
from li_ion_battery_p2d_init import initialize as init
an, cat, sep, solver_inputs, current, battery, anode, anode_s, elyte, cathode, cathode_s, conductor = init()

# Determine and return the state of an electrode region (electrode + 
#   electrolyte):
def set_state(offset, SV, ed, surf, el, conductor, ptr):
    # If this is the cathode, we need to set the molar composition:
    if ed.name == Inputs.cathode_phase:              
        ed.X = [SV[offset + ptr['X_ed'][-1]], 1 - SV[offset + ptr['X_ed'][-1]]]
        
    # Set the electrode phase electric potential:
    ed.electric_potential = SV[offset + ptr['Phi_ed']]
    conductor.electric_potential = SV[offset + ptr['Phi_ed']]
    
    # Set the electrolyte mole fraction and electric potential:
    el.X = SV[offset + ptr['X_k_elyte']]
    # We use this a few times, so let's save a local copy:
    Phi_el = SV[offset + ptr['Phi_dl']] + SV[offset + ptr['Phi_ed']]
    el.electric_potential = Phi_el
    
    # If this is the anode and we are considering an SEI phase, we need to 
    #   iterate to find the SEI interface potential such that the ionic current 
    #   across the SEI equals the Faradaic charge-transfer current:
    if ed.name == Inputs.anode_phase and Inputs.anode_SEI_flag:
        i = 0
        Phi_SEI_ed = np.linspace(SV[offset + ptr['Phi_dl']], 0, 1000)
        el.electric_potential = Phi_SEI_ed[i]
        sdot_Far = -surf.get_net_production_rates(el)[ptr['iFar']]
        i_Far = sdot_Far*ct.faraday*an.A_surf
        i_SEI = (Phi_el - Phi_SEI_ed[i])/an.R_SEI
        err = i_Far - i_SEI
        while err > 1/(10**Inputs.SEI_tol):
            el.electric_potential = Phi_SEI_ed[i]
            sdot_Far = -surf.get_net_production_rates(el)[ptr['iFar']]
            i_Far = sdot_Far*ct.faraday*an.A_surf
            i_SEI = (Phi_el - Phi_SEI_ed[i])/an.R_SEI
            err = i_Far - i_SEI
            i += 1
    
    state = {}
    state['sdot'] = surf.get_net_production_rates(el)
    state['phi_ed'] = SV[offset + ptr['Phi_ed']]
    state['phi_el'] = SV[offset + ptr['Phi_dl']] + SV[offset + ptr['Phi_ed']]
    state['rho_ed'] = ed.density_mole
    state['rho_el'] = el.density_mole
    state['X_k_el'] = SV[offset + ptr['X_k_elyte']]
    
    return state

"""====================================================================="""
# Determine and return the state of the electrolyte in the separator:
def set_state_sep(offset, SV, el, ptr):
    # Set the mole fraction:
    el.X = SV[offset + ptr['X_k_elyte']]

    # Set the electric potential:
    el.electric_potential = SV[offset + ptr['Phi']]
    
    state = {}
    state['phi_el'] = SV[offset + ptr['Phi']]
    state['rho_el'] = el.density_mole
    state['X_k_el'] = SV[offset + ptr['X_k_elyte']]
    
    return state

"""====================================================================="""
# Calculate the electrolyte species molar fluxes, according to dilute solution 
#   theory:
def dilute_flux(s1, s2, dyInv, ed, D_k, D_migr_k):
    # Faraday's constant, universal gas constant, temperature, and electrolyte 
    #   species charges:
    F = ct.faraday; R = ct.gas_constant; T = Inputs.T
    z_k = Inputs.z_k_elyte
    
    # Total molar concentration in electrolyte
    C_0 = (s2['rho_el'] + s1['rho_el'])/2.
    
    # Molar flux (kmol/m2/s)
    N_io_p = (- D_k*(s2['rho_el']*s2['X_k_el'] - s1['rho_el']*s1['X_k_el'])
              - D_migr_k*(z_k*F/R/T)*(s2['phi_el'] - s1['phi_el']))*dyInv

    # Corresponding ionic current:    
    i_io_p = np.dot(N_io_p, Inputs.z_k_elyte)*F
    
    return N_io_p, i_io_p

"""====================================================================="""
# DIffusive flux of intercalated Li:
def solid_flux(SV, offset, ptr, s1, ed):
    
    X_Li = SV[offset + ptr['X_ed']]
    DiffFlux = np.zeros([ed.nshells+1])
    DiffFlux[1:-1] = ed.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/ed.dr
    # Flux at the surface is equal to the production rate from the 
    #   intercalation reaction:
    DiffFlux[-1] = -s1['sdot'][ed.ptr['iFar']]/s1['rho_ed']
    
    return DiffFlux

# Initialize the requested plots:
def setup_plots(plt, rate_tag):
    # Charge-discharge.  Separate pane for each:
    if Inputs.plot_potential_profiles == 1:
        fig1, axes1 = plt.subplots(sharey="row", figsize=(10,6), nrows=1, 
                                   ncols = 2+(Inputs.flag_re_equil*Inputs.phi_time))
        plt.subplots_adjust(wspace = 0.15, hspace = 0.4)
        fig1.text(0.15, 0.8, rate_tag, fontsize=20, 
                  bbox=dict(facecolor='white', alpha=0.5))
    
    # Li intercalation fraction in cathode:
    if Inputs.plot_electrode_profiles == 1:
        nrows = Inputs.flag_cathode
        ncols = 2 + Inputs.flag_re_equil
        fig2, axes2 = plt.subplots(sharey="row", figsize=(10,9), nrows=nrows,
                                   ncols=ncols)
        plt.subplots_adjust(wspace=0.15, hspace=0.4)
        fig2.text(0.15, 0.8, rate_tag, fontsize=20, 
                  bbox=dict(facecolor='white', alpha=0.5))
    
    # Li+ electrolyte concentration at anode interface, in separator, and 
    #   throughout cathode thickness:
    if Inputs.plot_elyte_profiles == 1:
        nrows = Inputs.flag_anode + Inputs.flag_cathode + Inputs.flag_sep
        ncols = 2 + Inputs.flag_re_equil
        fig3, axes3 = plt.subplots(sharey="row", figsize=(10,9), nrows=nrows,
                                    ncols=ncols)
        plt.subplots_adjust(wspace=0.15, hspace=0.4)
        fig3.text(0.15, 0.8, rate_tag, fontsize=20, 
                  bbox=dict(facecolor='white', alpha=0.5))
        
    return fig1, axes1, fig2, axes2, fig3, axes3

