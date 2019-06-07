# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:47:07 2018

@author: Owner

"""

###############################################################################
#
#   Psuedo-2D Battery model
#
#       -D. Korff and S. DeCaluwe
#        Colorado School of Mines
#        2018-2019
#
###############################################################################

import numpy as np
import time
import importlib
import cantera as ct
from matplotlib import pyplot as plt

from assimulo.solvers import IDA
from assimulo.problem import Implicit_Problem
from assimulo.exception import TerminateSimulation

import li_ion_battery_p2d_functions
importlib.reload(li_ion_battery_p2d_functions)
from li_ion_battery_p2d_functions import setup_plots

import li_ion_battery_p2d_inputs
importlib.reload(li_ion_battery_p2d_inputs)
from li_ion_battery_p2d_inputs import Inputs

import li_ion_battery_p2d_init
importlib.reload(li_ion_battery_p2d_init)
from li_ion_battery_p2d_init import anode as an
from li_ion_battery_p2d_init import cathode as cat
from li_ion_battery_p2d_init import separator as sep
from li_ion_battery_p2d_init import solver_inputs, current

import li_ion_battery_p2d_post_process
importlib.reload(li_ion_battery_p2d_post_process)
from li_ion_battery_p2d_post_process import Label_Columns, tag_strings
from li_ion_battery_p2d_post_process import plot_potential, plot_electrode, plot_elyte
from li_ion_battery_p2d_post_process import plot_cap

import sys
sys.path.append('C:\\Users\\dkorff\\Research\\BatCan-repo\\functions')

import diffusion_coeffs
transport = getattr(diffusion_coeffs, Inputs.elyte_flux_model)

def main():
    
    SV_0 = solver_inputs.SV_0
    SV_dot_0 = np.zeros_like(SV_0)
    algvar = solver_inputs.algvar

    # Close any open pyplot objects:
    plt.close('all')
    
    atol = np.ones_like(SV_0)*1e-6
    rtol = 1e-4   

    # Start a timer:
    t_count = time.time()

    # Calculate the time span, which should be enough to charge or discharge fully
    #   (i.e. 3600 seconds divided by the C-rate):
    t_0 = 0
    t_f = 3600/Inputs.C_rate
    
    rate_tag = str(Inputs.C_rate)+"C"
    
    """----------Figures----------"""
    
    if Inputs.plot_profiles_flag:
        fig1, axes1, fig2, axes2, fig3, axes3 = setup_plots(plt, rate_tag)
        
    for cycle in np.arange(0, Inputs.n_cycles):
        """----------Equilibration----------"""
    
        # Equilibrate by integrating at zero current:
        print('\nEquilibrating...')
    
        # Create problem instance
        current.set_i_ext(0)
            
        battery_eq = li_ion(li_ion.res_fun, SV_0, SV_dot_0, t_0)
        battery_eq.external_event_detection = True
        battery_eq.algvar = algvar
    
        # Simulation parameters
        sim_eq = IDA(battery_eq)           # Create simulation instance
        sim_eq.atol = atol                   # Solver absolute tolerance
        sim_eq.rtol = rtol                  # Solver relative tolerance
        sim_eq.verbosity = 50
        sim_eq.make_consistent('IDA_YA_YDP_INIT')
    
        t_eq, SV_eq, SV_dot_eq = sim_eq.simulate(0.1*t_f)
    
        # Put solution into pandas dataframe with labeled columns
        SV_eq_df = Label_Columns(t_eq, SV_eq, an.npoints, sep.npoints, 
                                 cat.npoints)
    
        # Obtain tag strings for dataframe columns
        tags = tag_strings(SV_eq_df)
    
        print('Done equilibrating\n')
    
        """------------Charging-------------"""
        print('\nCharging...')
    
        # New initial conditions are the final equilibrium conditions
        t_0 = 0
        SV_0 = SV_eq[-1, :]
        SV_dot_0 = SV_dot_eq[-1 :]
    
        # Charge the battery
        current.set_i_ext(current.i_ext_set)
    
        # Create problem instance
        battery_ch = li_ion(li_ion.res_fun, SV_0, SV_dot_0, t_0)
        battery_ch.external_event_detection = True
        battery_ch.algvar = algvar
    
        # Simulation parameters
        sim_ch = IDA(battery_ch)
        sim_ch.atol = atol
        sim_ch.rtol = rtol
        sim_ch.verbosity = 50
        sim_ch.make_consistent('IDA_YA_YDP_INIT')
    
        t_ch, SV_ch, SV_dot_ch = sim_ch.simulate(t_f)
    
        SV_ch_df = Label_Columns(t_ch, SV_ch, an.npoints, 
                                     sep.npoints, cat.npoints)
        
        if Inputs.plot_potential_profiles == 1:
            plot_potential(tags['Phi_an'], tags['Phi_cat'], SV_ch_df, 
                      'Charging', 0, fig1, axes1)
            
        if Inputs.plot_electrode_profiles == 1:
            plot_electrode(tags['X_an'], tags['X_cat'], SV_ch_df, 
                           'Charging', 0, fig2, axes2)
        
        if Inputs.plot_elyte_profiles == 1:
            plot_elyte(tags['X_el_an'], tags['X_el_cat'], tags['X_el_sep'], SV_ch_df,
                       'Charging', 0, fig3, axes3)
    
        print('Done charging\n')
    
        """------------Re_equilibrating-------------"""
        
        if Inputs.flag_re_equil == 1:
            # New initial conditions are the final charge conditions
            SV_0 = SV_ch[-2, :]
            SV_dot_0 = SV_dot_ch[-2, :]
        
            # Equilibrate again. Note - this is a specific choice to reflect
            #   equilibration after the charging steps. We may want, at times, to
            #   simulate a situation where the battery is not equilibrated between
            #   charge and discharge, or is equilibrated for a shorter amount of time.
        
            print('\nRe-equilibrating...')
            
            current.set_i_ext(0)
            
            battery_req = li_ion(li_ion.res_fun, SV_0, SV_dot_0, t_0)
            battery_req.external_event_detection = True
            battery_req.algvar = algvar
        
            # Simulation parameters
            sim_req = IDA(battery_req)
            sim_req.atol = atol
            sim_req.rtol = rtol
            sim_req.verbosity = 50
            sim_req.make_consistent('IDA_YA_YDP_INIT')
        
            t_req, SV_req, SV_dot_req = sim_req.simulate(t_f)
        
            SV_req_df = Label_Columns(t_req, SV_req, an.npoints, sep.npoints, 
                                 cat.npoints)
            
            if Inputs.plot_potential_profiles*Inputs.phi_time == 1:
                plot_potential(tags['Phi_an'], tags['Phi_cat'], SV_req_df, 
                               'Re-equilibrating', 1, None, fig1, axes1)
            
            if Inputs.plot_electrode_profiles == 1:
                plot_electrode(tags['X_an'], tags['X_cat'], SV_req_df, 
                               'Re-equilibrating', 1, fig2, axes2)
                
            if Inputs.plot_elyte_profiles == 1:
                plot_elyte(tags['X_el_an'], tags['X_el_cat'], tags['X_el_sep'], SV_req_df,
                           'Re-equilibrating', 1, fig3, axes3)
        
            print('Done re-equilibrating\n')
        else:
            SV_req = SV_ch
            SV_dot_req = SV_dot_ch
            
            SV_req_df = SV_req
    
        """------------Discharging-------------"""
    
        print('\nDischarging...')
    
        SV_0 = SV_req[-1, :]
        SV_dot_0 = SV_dot_req[-1, :]
    
        current.set_i_ext(-current.i_ext_set)
    
        battery_dch = li_ion(li_ion.res_fun, SV_0, SV_dot_0, t_0)
        battery_dch.external_event_detection = True
        battery_dch.algvar = algvar
    
        # Simulation parameters
        sim_dch = IDA(battery_dch)
        sim_dch.atol = atol
        sim_dch.rtol = rtol
        sim_dch.verbosity = 50
        sim_dch.make_consistent('IDA_YA_YDP_INIT')
    
        t_dch, SV_dch, SV_dot_dch = sim_dch.simulate(t_f)
    
        SV_dch_df = Label_Columns(t_dch, SV_dch, an.npoints, sep.npoints, 
                                 cat.npoints)
        
        if Inputs.plot_potential_profiles == 1:
            plot_potential(tags['Phi_an'], tags['Phi_cat'], SV_dch_df, 
                      'Discharging', 1+(Inputs.flag_re_equil*Inputs.phi_time), fig1, axes1)
            
        if Inputs.plot_electrode_profiles == 1:
            plot_electrode(tags['X_an'], tags['X_cat'], SV_dch_df, 
                           'Discharging', 1+Inputs.flag_re_equil, fig2, axes2)
            
        if Inputs.plot_elyte_profiles == 1:
            plot_elyte(tags['X_el_an'], tags['X_el_cat'], tags['X_el_sep'], SV_dch_df,
                       'Discharging', 1+Inputs.flag_re_equil, fig3, axes3)
    
        print('Done discharging\n')
    
        """---------------------------------"""
        
        SV_0 = SV_dch[-1, :]
        SV_dot_0 = np.zeros_like(SV_0)
        
    # %% Plot capacity if flagged
        
        plt.show('all')
        
        plot_cap(SV_ch_df, SV_dch_df, rate_tag, current.i_ext_set,
                 Inputs.plot_cap_flag, tags)

    elapsed = time.time() - t_count
    print('t_cpu=', elapsed, '\n')
    plt.show()
    
    return SV_eq_df, SV_ch_df, SV_req_df, SV_dch_df

"============================================================================="
"============================================================================="
"============================================================================="

import li_ion_battery_p2d_init
importlib.reload(li_ion_battery_p2d_init)
from li_ion_battery_p2d_init import battery as bat
from li_ion_battery_p2d_init import anode_obj as anode
from li_ion_battery_p2d_init import anode_surf_obj as anode_s
from li_ion_battery_p2d_init import elyte_obj as elyte
from li_ion_battery_p2d_init import cathode_obj as cathode
from li_ion_battery_p2d_init import cathode_surf_obj as cathode_s
from li_ion_battery_p2d_init import conductor_obj as conductor

import li_ion_battery_p2d_functions
importlib.reload(li_ion_battery_p2d_functions)
from li_ion_battery_p2d_functions import set_state
from li_ion_battery_p2d_functions import set_state_sep
from li_ion_battery_p2d_functions import dilute_flux as elyte_flux
from li_ion_battery_p2d_functions import solid_flux

class li_ion(Implicit_Problem):
    def res_fun(t, SV, SV_dot):
        """================================================================="""
        """==========================INITIALIZE============================="""
        offsets = an.offsets; F = ct.faraday; params = bat.cst_params
        
        nSV = len(SV)
        res = np.zeros([nSV])
        i_ext = current.get_i_ext()
        
#        elyte_model = solver_inputs.elyte_model

# %%
        """================================================================="""
        """============================ANODE================================"""
        # Looking at node 0, CC boundary "outlet" conditions
        j = 0; offset = int(offsets[j]); ptr = an.ptr

        N_io_p = 0; i_io_p = 0; i_el_p = i_ext
        
        s1 = {}; s2 = {}
                
        # Shift forward to node 1, j=0, to set FIRST node conditions
        s2 = set_state(offset, SV, anode, anode_s, elyte, conductor, ptr)
        
        # Diffusive flux scaling factors
        k = np.arange(0, an.nshells+1)/an.nshells
                        
# %%
        """============================ANODE================================"""
        """INTERIOR NODES"""
        for j in np.arange(1, an.npoints):
            # Save previous node outlet conditions as new inlet conditions
            N_io_m = N_io_p
            i_io_m = i_io_p
            i_el_m = i_el_p
            s1 = s2

            # Shift forward to NEXT node
            offset = int(offsets[j])

            s2 = set_state(offset, SV, anode, anode_s, elyte, conductor, ptr)

            # Shift back to THIS node, set THIS node outlet conditions
            offset = int(offsets[j - 1])

            i_el_p = an.sigma_eff_ed*(s1['phi_ed'] - s2['phi_ed'])*an.dyInv
            
            transport.Dk_el_0 = an.D_el_eff
            transport.C_k = (s2['X_k_el']*s2['rho_el'] 
                           + s1['X_k_el']*s1['rho_el'])/2.
            transport.ptr_el = bat.ptr_el
            transport.z_k = Inputs.z_k_elyte
            transport.rho_bar = elyte.density_mole
            D_k, D_k_migr = transport.coeffs(params)
            
            N_io_p, i_io_p = elyte_flux(s1, s2, an.dyInv, an, D_k, D_k_migr)

            i_Far_1 = -s1['sdot'][ptr['iFar']]*F*an.A_surf/an.dyInv

#            DiffFlux = solid_flux(SV, offset, ptr, s1, an)
            X_Li = SV[offset + ptr['X_ed']]
            DiffFlux = np.zeros([an.nshells+1])
            DiffFlux[1:-1] = an.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/an.dr
            DiffFlux[-1] = -s1['sdot'][an.ptr['iFar']]/s1['rho_ed']

            """Calculate the change in X_LiC6 in the particle interior."""
            res[offset + an.ptr['X_ed']] = (SV_dot[offset + ptr['X_ed']]
            - ((DiffFlux[1:]*k[1:]**2 - DiffFlux[0:-1]*k[0:-1]**2)
            * an.A_surf/an.eps_ed/an.V_shell))

            """Change in electrolyte_composition"""
            res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
            - (((N_io_m - N_io_p)*an.dyInv + s1['sdot']*an.A_surf)
            /s1['rho_el']/an.eps_elyte))

            """Double-layer voltage"""
            res[offset + ptr['Phi_dl']] = (SV_dot[offset + ptr['Phi_dl']]
            - (-i_Far_1 + i_io_m - i_io_p)*an.dyInv/an.C_dl/an.A_surf)

            """Algebraic equation for ANODE electric potential boundary condition"""
            res[offset + ptr['Phi_ed']] = (i_el_m - i_el_p + i_io_m - i_io_p)
            
# %%
        """============================ANODE================================"""
        """Separator boundary"""
        # Save previous node outlet conditions as new inlet conditions
        N_io_m = N_io_p
        i_io_m = i_io_p
        i_el_m = i_el_p
        s1 = s2
     
        # Shift forward to NEXT node, first separator node (j=0)
        j = 0; offset = int(sep.offsets[j])

        s2 = set_state_sep(offset, SV, elyte, sep.ptr)

        # Shift back to THIS node, set THIS node outlet conditions
        i_el_p = 0

        # Set j to final ANODE node
        j = an.npoints-1; offset = int(an.offsets[j])

        i_Far_1 = -s1['sdot'][ptr['iFar']]*F*an.A_surf/an.dyInv
        
        dyInv_boundary = 1/(0.5*(1/an.dyInv + 1/sep.dyInv))
        
        transport.C_k = (s2['X_k_el']*s2['rho_el'] 
                           + s1['X_k_el']*s1['rho_el'])/2.
        transport.rho_bar = elyte.density_mole
        D_k, D_k_migr = transport.coeffs(params)
            
        N_io_p, i_io_p = elyte_flux(s1, s2, dyInv_boundary, an, D_k, D_k_migr)
        
#        DiffFlux = solid_flux(SV, offset, ptr, s1, an)
        X_Li = SV[offset + ptr['X_ed']]
        DiffFlux = np.zeros([an.nshells+1])
        DiffFlux[1:-1] = an.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/an.dr
        DiffFlux[-1] = -s1['sdot'][an.ptr['iFar']]/s1['rho_ed']
    
        """Calculate the change in X_LiC6 in the particle interior."""
        res[offset + ptr['X_ed']] = (SV_dot[offset + ptr['X_ed']]
        - ((DiffFlux[1:]*k[1:]**2 - DiffFlux[0:-1]*k[0:-1]**2)
        * an.A_surf/an.eps_ed/an.V_shell))

        """Change in electrolyte_composition"""
        res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
        - (((N_io_m - N_io_p)*an.dyInv + s1['sdot']*an.A_surf)
        /s1['rho_el']/an.eps_elyte))

        """Double-layer voltage"""
        res[offset + ptr['Phi_dl']] = (SV_dot[offset + ptr['Phi_dl']]
        - (-i_Far_1 + i_io_m - i_io_p)*an.dyInv/an.C_dl/an.A_surf)

        """Algebraic equation for ANODE electric potential boundary condition"""
        res[offset + ptr['Phi_ed']] = SV[an.ptr['Phi_ed']]
        
# %%
        """================================================================="""
        """==========================SEPARATOR=============================="""
        offsets = sep.offsets; ptr = sep.ptr
        
        for j in np.arange(1, sep.npoints):
            # Save previous node outlet conditions as new inlet conditions
            i_io_m = i_io_p
            N_io_m = N_io_p
            s1 = s2
            
            # Set NEXT separator node conditions
            offset = int(offsets[j])
            
            s2 = set_state_sep(offset, SV, elyte, ptr)
            
            # Shift back to THIS node
            offset = int(sep.offsets[j-1])
            
            transport.C_k = (s2['X_k_el']*s2['rho_el'] 
                           + s1['X_k_el']*s1['rho_el'])/2.
            transport.rho_bar = elyte.density_mole
            D_k, D_k_migr = transport.coeffs(params)
            
            N_io_p, i_io_p = elyte_flux(s1, s2, sep.dyInv, sep, D_k, D_k_migr)
        
            """Change in electrolyte composition"""
            res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
            - (((N_io_m - N_io_p)*sep.dyInv)/s1['rho_el']/sep.eps_elyte))
            
            """Algebraic equation for electrolyte potential"""
            res[offset + ptr['Phi']] = i_io_m - i_io_p
            
        """==========================SEPARATOR=============================="""
        """Cathode boundary"""
        
        i_io_m = i_io_p
        N_io_m = N_io_p
        s1 = s2
        
        # Shift forward to NEXT node, first cathode node (j=0)
        j = 0; offset = int(cat.offsets[j])

        s2 = set_state(offset, SV, cathode, cathode_s, elyte, conductor, cat.ptr)
        
        # Shift to final separator node
        j = sep.npoints-1; offset = int(offsets[j])
        
        i_el_p = 0
        
        dyInv_boundary = 1/(0.5*(1/cat.dyInv + 1/sep.dyInv))
        
        transport.C_k = (s2['X_k_el']*s2['rho_el'] 
                           + s1['X_k_el']*s1['rho_el'])/2.
        transport.rho_bar = elyte.density_mole
        D_k, D_k_migr = transport.coeffs(params)

        N_io_p, i_io_p = elyte_flux(s1, s2, dyInv_boundary, sep, D_k, D_k_migr)
                
        """Change in electrolyte composition"""
        res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
        - (((N_io_m - N_io_p)*sep.dyInv)/s1['rho_el']/sep.eps_elyte))
      
        """Algebraic equation for electrolyte potential"""
        res[offset + ptr['Phi']] = i_io_m - i_io_p

# %%
        """================================================================="""
        """===========================CATHODE==============================="""
        offsets = cat.offsets; ptr = cat.ptr
        
        k = np.arange(0, cat.nshells+1)/cat.nshells
                
        """=========================CATHODE============================="""
        """INTERIOR NODES"""
        
        for j in np.arange(1, cat.npoints):
            # Save previous node outlet conditions as new inlet conditions
            N_io_m = N_io_p
            i_io_m = i_io_p
            i_el_m = i_el_p
            s1 = s2
            
            # Shift forward to NEXT node
            offset = int(offsets[j])
            
            s2 = set_state(offset, SV, cathode, cathode_s, elyte, conductor, ptr)
            
            # Shift back to THIS node, set THIS node outlet conditions
            offset = int(offsets[j-1])
            
            i_el_p = cat.sigma_eff_ed*(s1['phi_ed'] - s2['phi_ed'])*cat.dyInv
            
            transport.C_k = (s2['X_k_el']*s2['rho_el'] 
                           + s1['X_k_el']*s1['rho_el'])/2.
            transport.rho_bar = elyte.density_mole
            D_k, D_k_migr = transport.coeffs(params)
            
            N_io_p, i_io_p = elyte_flux(s1, s2, cat.dyInv, cat, D_k, D_k_migr)
            
            i_Far_1 = -s1['sdot'][ptr['iFar']]*F*cat.A_surf/cat.dyInv
            
#            DiffFlux = solid_flux(SV, offset, ptr, s1, cat)
            X_Li = SV[offset + ptr['X_ed']]
            DiffFlux = np.zeros([cat.nshells+1])
            DiffFlux[1:-1] = cat.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/cat.dr
            DiffFlux[-1] = -s1['sdot'][cat.ptr['iFar']]/s1['rho_ed']

            """Calculate the change in X_LiCoO2 in the particle interior"""
            res[offset + ptr['X_ed']] = (SV_dot[offset + ptr['X_ed']]
            - ((DiffFlux[1:]*k[1:]**2 - DiffFlux[0:-1]*k[0:-1]**2)
            *cat.A_surf/cat.eps_ed/cat.V_shell))
            
            """Change in electrolyte composition"""
            res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
            - (((N_io_m - N_io_p)*cat.dyInv + s1['sdot']*cat.A_surf)
            /s1['rho_el']/cat.eps_elyte))
            
            """Double-layer voltage"""
            res[offset + ptr['Phi_dl']] = (SV_dot[offset + ptr['Phi_dl']]
            - (-i_Far_1 + i_io_m - i_io_p)*cat.dyInv/cat.C_dl/cat.A_surf)
            
            """Algebraic equation for CATHODE electric potential"""
            res[offset + ptr['Phi_ed']] = (i_el_m - i_el_p + i_io_m - i_io_p)
        
# %%
        """=========================CATHODE============================="""
        """current collector boundary"""
        N_io_m = N_io_p
        i_io_m = i_io_p
        i_el_m = i_el_p
        s1 = s2
        
        # FINAL node
        j = cat.npoints-1; offset = int(offsets[j])
        
        i_io_p = 0
        N_io_p = 0
        i_el_p = i_ext
        
        i_Far_1 = -s1['sdot'][ptr['iFar']]*F*cat.A_surf/cat.dyInv
        
#        DiffFlux = solid_flux(SV, offset, ptr, s1, cat)
        X_Li = SV[offset + ptr['X_ed']]
        DiffFlux = np.zeros([cat.nshells+1])
        DiffFlux[1:-1] = cat.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/cat.dr
        DiffFlux[-1] = -s1['sdot'][cat.ptr['iFar']]/s1['rho_ed']
                        
        """Calculate the change in X_LiCoO2 in the particle interior"""
        res[offset + ptr['X_ed']] = (SV_dot[offset + ptr['X_ed']]
        - ((DiffFlux[1:]*k[1:]**2 - DiffFlux[0:-1]*k[0:-1]**2)
        *cat.A_surf/cat.eps_ed/cat.V_shell))
        
        """Change in electrolyte composition"""
        res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
        - (((N_io_m - N_io_p)*cat.dyInv + s1['sdot']*cat.A_surf)
        /s1['rho_el']/cat.eps_elyte))
        
        """Double-layer voltage"""
        res[offset + ptr['Phi_dl']] = (SV_dot[offset + ptr['Phi_dl']]
        - (-i_Far_1 + i_io_m - i_io_p)*cat.dyInv/cat.C_dl/cat.A_surf)
        
        """Algebraic equation for CATHODE electric potential"""
        res[offset + ptr['Phi_ed']] = (i_el_m - i_el_p + i_io_m - i_io_p)
                        
        return res

# %%
    """====================================================================="""
    """==========================Solver Functions==========================="""
    """====================================================================="""

    def state_events(self, t, y, yd, sw):
        
        # Anode events
        event1 = np.zeros([an.npoints])
        event2 = np.zeros([an.npoints])
        event3 = np.zeros([an.npoints*an.nshells])
        event4 = np.zeros([an.npoints*an.nshells])
        
        event1 = y[an.ptr_vec['Phi_dl']]
        event2 = 5 - y[an.ptr_vec['Phi_dl']]
        event3 = an.X_Li_max - y[an.ptr_vec['X_ed']]
        event4 = y[an.ptr_vec['X_ed']] - an.X_Li_min
            
        # Cathode events
        event5 = np.zeros([cat.npoints])
        event6 = np.zeros([cat.npoints])
        event7 = np.zeros([cat.npoints*cat.nshells])
        event8 = np.zeros([cat.npoints*cat.nshells])
        
        event5 = y[cat.ptr_vec['Phi_dl']]
        event6 = 5 - y[cat.ptr_vec['Phi_ed']]
        event7 = cat.X_Li_max - y[cat.ptr_vec['X_ed']]
        event8 = y[cat.ptr_vec['X_ed']] - cat.X_Li_min
               
        # Electrolyte events
        event9  = np.zeros([an.npoints*elyte.n_species])
        event10 = np.zeros([an.npoints*elyte.n_species])
        event11 = np.zeros([cat.npoints*elyte.n_species])
        event12 = np.zeros([cat.npoints*elyte.n_species])
        
        event9  = 1 - y[an.ptr_vec['X_k_elyte']]
        event10 = y[an.ptr_vec['X_k_elyte']]
        event11 = 1 - y[cat.ptr_vec['X_k_elyte']]
        event12 = y[cat.ptr_vec['X_k_elyte']]

        # Concatenate events into one array
        events = np.concatenate((event1, event2, event3, event4, 
                                 event5, event6, event7, event8, 
                                 event9, event10, event11, event12))

        return events

    """====================================================================="""

    def handle_event(self, solver, event_info):
        
        state_info = event_info[0] #We are only interested in state events info
        event_ptr = {}
        event_ptr['An_phi1'] = an.npoints
        event_ptr['An_phi2'] = event_ptr['An_phi1'] + an.npoints
        event_ptr['An_Xed1'] = event_ptr['An_phi2'] + an.npoints*an.nshells
        event_ptr['An_Xed2'] = event_ptr['An_Xed1'] + an.npoints*an.nshells
        
        event_ptr['Cat_phi1'] = event_ptr['An_Xed2'] + cat.npoints
        event_ptr['Cat_phi2'] = event_ptr['Cat_phi1'] + cat.npoints
        event_ptr['Cat_Xed1'] = event_ptr['Cat_phi2'] + cat.npoints*cat.nshells
        event_ptr['Cat_Xed2'] = event_ptr['Cat_Xed1'] + cat.npoints*cat.nshells
        
        event_ptr['An_el1'] = event_ptr['Cat_Xed2'] + an.npoints*elyte.n_species
        event_ptr['An_el2'] = event_ptr['An_el1'] + an.npoints*elyte.n_species
        event_ptr['Cat_el1'] = event_ptr['An_el2'] + cat.npoints*elyte.n_species
        event_ptr['Cat_el2'] = event_ptr['Cat_el1'] + cat.npoints*elyte.n_species
        
        if any(state_info[0:event_ptr['An_phi1']]):
            print('Cutoff: anode double-layer flipped sign')
            raise TerminateSimulation
        elif any(state_info[event_ptr['An_phi1']:event_ptr['An_phi2']]):
            print('Cutoff: anode double-layer blew up')
            raise TerminateSimulation
        elif any(state_info[event_ptr['An_phi2']:event_ptr['An_Xed1']]):
            print('Cutoff: Anode shell fully lithiated')
            raise TerminateSimulation
        elif any(state_info[event_ptr['An_Xed1']:event_ptr['An_Xed2']]):
            print('Cutoff: Anode shell fully de-lithiated')
            raise TerminateSimulation
        elif any(state_info[event_ptr['An_Xed2']:event_ptr['Cat_phi1']]):
            print('Cutoff: Cathode double layer flipped sign')
            raise TerminateSimulation
        elif any(state_info[event_ptr['Cat_phi1']:event_ptr['Cat_phi2']]):
            print('Cutoff: Cell potential went over 5 V')
            raise TerminateSimulation
        elif any(state_info[event_ptr['Cat_phi2']:event_ptr['Cat_Xed1']]):
            print('Cutoff: Cathode shell fully lithiated')
            raise TerminateSimulation
        elif any(state_info[event_ptr['Cat_Xed1']:event_ptr['Cat_Xed2']]):
            print('Cutoff: Cathode shell fully de-lithiated')
            raise TerminateSimulation
        elif any(state_info[event_ptr['An_el1']:event_ptr['An_el2']]):
            print('Cutoff: Li+ in electrolyte for anode depleted')
            raise TerminateSimulation
        elif any(state_info[event_ptr['Cat_el1']:event_ptr['Cat_el2']]):
            print('Cutoff: Li+ in electrolyte for cathode depleted')
            raise TerminateSimulation

    """====================================================================="""

if __name__ == "__main__":
    SV_eq_df, SV_charge_df, SV_req_df, SV_discharge_df = main()
