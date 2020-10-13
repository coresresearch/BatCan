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
import os
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
from li_ion_battery_p2d_inputs import Inputs

from li_ion_battery_p2d_init import initialize as init
(an, ca, sep, solver_inputs, current, bat, anode, anode_s, elyte, cathode, 
    cathode_s, conductor) = init()

import li_ion_battery_p2d_post_process
importlib.reload(li_ion_battery_p2d_post_process)
from li_ion_battery_p2d_post_process import Label_Columns, tag_strings
from li_ion_battery_p2d_post_process import plot_potential, plot_electrode, plot_elyte
from li_ion_battery_p2d_post_process import plot_cap

import sys
sys.path.append('../functions')


def main(kinetics=None, H_ca=None, i_ext=None, C_rate=None):
    
    SV_0 = solver_inputs.SV_0
    SV_dot_0 = np.zeros_like(SV_0)
    algvar = solver_inputs.algvar

    # Close any open pyplot objects:
    plt.close('all')
    
    atol = np.ones_like(SV_0)*1e-8
    rtol = 1e-7  

    # Start a timer:
    t_count = time.time()
    if hasattr(Inputs, 'i_ext'):
        Inputs.C_rate = -current.i_ext_set/min(an.oneC, ca.oneC)

    print("Current is {:.2f} mA/cm2.".format(-current.i_ext_set*1000/10000))
    
    # Calculate the time span, which should be enough to charge or discharge 
    #   fully (i.e. 3600 seconds divided by the C-rate):
    t_0 = 0
    t_f = 3600/Inputs.C_rate
    
    rate_tag = str(round(Inputs.C_rate,1))+"C"
    
    """----------Figures----------"""
    
    if Inputs.plot_profiles_flag:
        fig1, axes1, fig2, axes2, fig3, axes3 = setup_plots(plt, rate_tag)
        
    for cycle in np.arange(0, Inputs.n_cycles):
        """----------Equilibration----------"""
    
        # Equilibrate by integrating at zero current. This is necessary for an 
        #   internally consistent initial condition:
        print('\nEquilibrating...')

        # Set the current to zero:
        current.set_i_ext(0)
            
        # Create problem instance
        battery_eq = li_ion(li_ion.res_fun, SV_0, SV_dot_0, t_0)
        battery_eq.external_event_detection = True
        battery_eq.algvar = algvar
    
        # Simulation parameters
        sim_eq = IDA(battery_eq)           # Create simulation instance
        sim_eq.atol = atol                   # Solver absolute tolerance
        sim_eq.rtol = rtol                  # Solver relative tolerance
        sim_eq.verbosity = 50
        sim_eq.make_consistent('IDA_YA_YDP_INIT')
    
        # Run the simulation:
        t_eq, SV_eq, SV_dot_eq = sim_eq.simulate(0.1*t_f)
    
        # Put solution into pandas dataframe with labeled columns
        SV_eq_df = Label_Columns(t_eq, SV_eq, an.npoints, sep.npoints, 
                                 ca.npoints)
    
        # Obtain tag strings for dataframe columns
        tags = tag_strings(SV_eq_df)
            
        print('Done equilibrating\n')
    
        """------------Charging-------------"""
        print('\nCharging...')
    
        # New initial conditions are the equilibrium conditions from above:
        t_0 = 0
        SV_0 = SV_eq[-1, :]
        SV_dot_0 = SV_dot_eq[-1 :]
    
        # Set the current to the user input value:
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
    
        # Run the simulation:
        t_ch, SV_ch, SV_dot_ch = sim_ch.simulate(t_f)
    
        # Put solution into pandas dataframe with labeled columns
        SV_ch_df = Label_Columns(t_ch, SV_ch, an.npoints, 
                                     sep.npoints, ca.npoints)
        
        # Add results to the user-requested plots:
        if Inputs.plot_potential_profiles == 1:
            plot_potential(tags['Phi_an'], tags['Phi_ca'], SV_ch_df, 
                      'Charging', 0, fig1, axes1)
            
        if Inputs.plot_electrode_profiles == 1:
            plot_electrode(tags['X_ca'], SV_ch_df, 'Charging', 0, fig2, axes2)
        
        if Inputs.plot_elyte_profiles == 1:
            plot_elyte(tags['X_el_an'], tags['X_el_ca'], tags['X_el_sep'], SV_ch_df, 'Charging', 0, fig3, axes3)
    
        print('Done charging\n')
        
        # Before running the discharge, see if any of the cathode conditions 
        #   are at the 'termination' condition, at the end of charging.  This 
        #   may cause the discharge simualtion to terminate prematurely, so we 
        #   move them sufficiently far from the termination condition, before 
        #   continuing:
        for i in np.arange(0, len(ca.ptr_vec['X_ed'])):
            ptr = ca.ptr_vec['X_ed'][i]
            if (SV_ch[-1, ptr] - Inputs.Li_ca_min) < Inputs.Li_ca_min/10:
                SV_ch[-1, ptr] += Inputs.Li_ca_min/10
    
        """------------Re_equilibrating-------------"""
        
        if Inputs.flag_re_equil == 1:
            # New initial conditions are the final charge conditions
            SV_0 = SV_ch[-2, :]
            SV_dot_0 = SV_dot_ch[-2, :]
        
            # Equilibrate again. Note - this is a specific choice to reflect
            #   equilibration after the charging steps. We may want, at times, 
            #   to simulate a protocol where the battery is not equilibrated 
            #   between charge and discharge, or is equilibrated for a shorter 
            #   amount of time.
        
            print('\nRe-equilibrating...')
            
            # Set the current back to zero:
            current.set_i_ext(0)
            
            # Create a problem instance:
            battery_req = li_ion(li_ion.res_fun, SV_0, SV_dot_0, t_0)
            battery_req.external_event_detection = True
            battery_req.algvar = algvar
        
            # Simulation parameters
            sim_req = IDA(battery_req)
            sim_req.atol = atol
            sim_req.rtol = rtol
            sim_req.verbosity = 50
            sim_req.make_consistent('IDA_YA_YDP_INIT')
        
            # Run the simulation:
            t_req, SV_req, SV_dot_req = sim_req.simulate(Inputs.equil_time*t_f)
        
            # Put solution into pandas dataframe with labeled columns
            SV_req_df = Label_Columns(t_req, SV_req, an.npoints, sep.npoints, 
                                 ca.npoints)
            
            # Add results to the user-requested plots:
            if Inputs.plot_potential_profiles*Inputs.phi_time == 1:
                plot_potential(tags['Phi_an'], tags['Phi_ca'], SV_req_df, 
                               'Re-equilibrating', 1, None, fig1, axes1)
            
            if Inputs.plot_electrode_profiles == 1:
                plot_electrode(tags['X_ca'], SV_req_df, 
                               'Re-equilibrating', 1, fig2, axes2)
                
            if Inputs.plot_elyte_profiles == 1:
                plot_elyte(tags['X_el_an'], tags['X_el_ca'], tags['X_el_sep'], SV_req_df,
                           'Re-equilibrating', 1, fig3, axes3)
        
            print('Done re-equilibrating\n')
        else:
            # If re-equilibration is not run, copy the charging results to the 
            #   're-equilibrated' results, which are used to create the initial 
            #   conditions for the discharge simulation:
            SV_req = SV_ch
            SV_dot_req = SV_dot_ch
            SV_req_df = SV_req
    
        """------------Discharging-------------"""
    
        print('\nDischarging...')
    
        # Load the initial conditions from the terminal condition of the 
        #   previous run:
        SV_0 = SV_req[-1, :]
        SV_dot_0 = SV_dot_req[-1, :]
    
        # Set the current density to the user-provided input:
        current.set_i_ext(-current.i_ext_set)
    
        # Create a problem instance:
        battery_dch = li_ion(li_ion.res_fun, SV_0, SV_dot_0, t_0)
        battery_dch.external_event_detection = True
        battery_dch.algvar = algvar
    
        # Simulation parameters
        sim_dch = IDA(battery_dch)
        sim_dch.atol = atol
        sim_dch.rtol = rtol
        sim_dch.verbosity = 50
        sim_dch.make_consistent('IDA_YA_YDP_INIT')

        # Run the simulation:
        t_dch, SV_dch, SV_dot_dch = sim_dch.simulate(t_f)
    
        # Put solution into pandas dataframe with labeled columns
        SV_dch_df = Label_Columns(t_dch, SV_dch, an.npoints, sep.npoints, 
                                 ca.npoints)
        
        # Add results to the user-requested plots:
        if Inputs.plot_potential_profiles == 1:
            plot_potential(tags['Phi_an'], tags['Phi_ca'], SV_dch_df, 
                'Discharging', 1+(Inputs.flag_re_equil*Inputs.phi_time), fig1, axes1)
            
        if Inputs.plot_electrode_profiles == 1:
            plot_electrode(tags['X_ca'], SV_dch_df, 
                'Discharging', 1+Inputs.flag_re_equil, fig2, axes2)
            
        if Inputs.plot_elyte_profiles == 1:
            plot_elyte(tags['X_el_an'], tags['X_el_ca'], tags['X_el_sep'],
                SV_dch_df, 'Discharging', 1+Inputs.flag_re_equil, fig3, axes3)
    
        print('Done discharging\n')

        # Calculate the time it took to run the simulations:
        elapsed = time.time() - t_count
        print('t_cpu=', elapsed, '\n')
    
        """---------------------------------"""
        
        # Save the final discharge conditions as the initial conditions for the 
        #   next cycle:
        SV_0 = SV_dch[-1, :]
        SV_dot_0 = np.zeros_like(SV_0)
        
        # Plot capacity if flagged
        fig1.savefig(Inputs.save_path+Inputs.save_name+'_Potentials.png',dpi=350)
        fig2.savefig(Inputs.save_path+Inputs.save_name+'_ElectrodeComposition.png',dpi=350)
        fig3.savefig(Inputs.save_path+Inputs.save_name+'_ElyteConcentration.png',dpi=350)
        
        plot_cap(SV_ch_df, SV_dch_df, rate_tag, current.i_ext_set,
                 Inputs.plot_cap_flag, tags)
        
    # Convert time to capacity and export
    if Inputs.cap_method == 'areal':
        file_name_dch = (Inputs.save_path+'dch_'+Inputs.save_name
            +'_AreaCap_discharge_data.csv')
        file_name_ch = (Inputs.save_path+'ch_'+Inputs.save_name
            +'_AreaCap_charge_data.csv')
        SV_dch = SV_dch_df.copy()
        SV_dch.loc[:, 'Time'] *= -current.i_ext_amp/3600
        SV_ch = SV_ch_df.copy()
        SV_ch.loc[:, 'Time'] *= -current.i_ext_amp/3600
        SV_dch.to_csv(file_name_dch, index=False, header=True)
        SV_ch.to_csv(file_name_ch, index=False, header=True)
    elif Inputs.cap_method == 'grav' and Inputs.grav_cap_method == 'cathode':
        file_name_dch = (Inputs.save_path+'dch_'+Inputs.save_name
            +'_GravCatCap_discharge_data.csv')
        file_name_ch = (Inputs.save_path+'ch_'+Inputs.save_name
            +'_GravCatCap_charge_data.csv')
        SV_dch = SV_dch_df.copy()
        SV_dch.loc[:, 'Time'] *= -current.i_ext_amp/3600/ca.rho_ed/ca.H/ca.eps_ed
        SV_ch = SV_ch_df.copy()
        SV_ch.loc[:, 'Time'] *= -current.i_ext_amp/3600/ca.rho_ed/ca.H/ca.eps_ed
        SV_dch.to_csv(file_name_dch, index=False, header=True)
        SV_ch.to_csv(file_name_ch, index=False, header=True)
    elif Inputs.cap_method == 'grav' and Inputs.grav_cap_method == 'cell':
        file_name_dch = (Inputs.save_path+'dch_'+Inputs.save_name
            +'_GravCellCap_discharge_data.csv')
        file_name_ch = (Inputs.save_path+'ch_'+Inputs.save_name
            +'_GravCellCap_charge_data.csv')
        SV_dch = SV_dch_df.copy()
        SV_dch.loc[:, 'Time'] *= -current.i_ext_amp/3600/bat.mass
        SV_ch = SV_ch_df.copy()
        SV_ch.loc[:, 'Time'] *= -current.i_ext_amp/3600/bat.mass
        SV_dch.to_csv(file_name_dch, index=False, header=True)
        SV_ch.to_csv(file_name_ch, index=False, header=True)


    from shutil import copy2
    """Copy Inputs and the Cantera Input file to the outputs:"""
    "-------------------------------------------------------------------------"
    copy2('li_ion_battery_p2d_inputs.py', Inputs.save_path)
    copy2(Inputs.canterafile, Inputs.save_path)

    plt.show()
    
    return SV_eq_df, SV_ch_df, SV_req_df, SV_dch_df

"============================================================================="
"============================================================================="
"============================================================================="

import li_ion_battery_p2d_init
importlib.reload(li_ion_battery_p2d_init)

import li_ion_battery_p2d_functions
importlib.reload(li_ion_battery_p2d_functions)
from li_ion_battery_p2d_functions import set_state
from li_ion_battery_p2d_functions import set_state_sep
from li_ion_battery_p2d_functions import dilute_flux as elyte_flux
from li_ion_battery_p2d_functions import solid_flux

class li_ion(Implicit_Problem):
    def res_fun(t, SV, SV_dot):

        import diffusion_coeffs
        transport = getattr(diffusion_coeffs, Inputs.elyte_flux_model)

        """================================================================="""
        """==========================INITIALIZE============================="""
        offsets = an.offsets; F = ct.faraday
        
        nSV = len(SV)
        res = np.zeros([nSV])
        i_ext = current.get_i_ext()

        """================================================================="""
        """============================ANODE================================"""
        # Looking at node 0, CC boundary "outlet" conditions
        j = 0; offset = int(offsets[j]); ptr = an.ptr

        N_elyte_p = 0; i_io_p = 0; i_el_p = i_ext
        
        s1 = {}; s2 = {}
                
        # Shift forward to node 1, j=0, to set FIRST node conditions
        s2 = set_state(offset, SV, anode, anode_s, elyte, conductor, ptr)
        
        # Diffusive flux scaling factors
        transport.ptr_el = bat.ptr_el
        transport.z_k = Inputs.z_k_elyte
        transport.Dk_el_0 = an.D_el_eff
        transport.params = bat.cst_params
        transport.T = Inputs.T
        transport.T_0 = Inputs.T

        """============================ANODE================================"""
        """Current collector boundary"""
        # Save previous node outlet conditions as new inlet conditions
        N_elyte_m = 0
        i_io_m = 0
        i_el_m = i_ext
        s1 = s2
     
        # Shift forward to NEXT node, first separator node (j=0)
        j = 0; offset = int(sep.offsets[j])

        s2 = set_state_sep(offset, SV, elyte, sep.ptr)

        # Shift back to THIS node, set THIS node outlet conditions
        i_el_p = 0

        # Set j to final ANODE node
        j = an.npoints-1; offset = int(an.offsets[j])

        i_Far_1 = -s1['sdot'][ptr['iFar']]*F*an.A_surf
        
        # Distance between node centers:
        dyInv_boundary = 1/(0.5*(1/an.dyInv_el + 1/sep.dyInv))

        # Weighting factors for average interface properties:
        w1 = sep.dy/(an.dy_el + sep.dy); w2 = an.dy_el/(an.dy_el + sep.dy)
        # Average species concentrations at the interface:
        #TODO #12
        transport.C_k = (s2['X_k_el']*s2['rho_el']*w2 
                           + s1['X_k_el']*s1['rho_el']*w1)
        # Average total concentration:
        transport.rho_bar = (s2['rho_el']*w2 + s1['rho_el']*w1)
        # Diffusion and migration coefficients:
        D_k, D_k_migr = transport.coeffs()
            
        # Electrolyte species fluxes and associated ionic current:
        N_elyte_p, i_io_p = elyte_flux(s1, s2, dyInv_boundary, an, D_k, 
            D_k_migr)

        # Double layer current:
        i_dl = -i_Far_1 - i_el_m + i_el_p
        # Convert double layer current to an equivalent species production rate.
        #TODO #11
        R_dl = np.array((0, 0, -i_dl/an.dy_el/F, 0))

        """Change in electrolyte_composition"""
        # The separator porosity & dyInv_el are used here as this equation 
        #   governs electrolyte in the separator contained in a small volume at 
        #   the anode interface.
        res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
            - (((N_elyte_m - N_elyte_p)*an.dyInv_el + s1['sdot']*an.A_surf*an.
            dyInv_el + R_dl)/s1['rho_el']/sep.eps_elyte))

        """Double-layer voltage"""
        res[offset + ptr['Phi_dl']] = (SV_dot[offset + ptr['Phi_dl']]
            - (i_dl)/an.C_dl/an.A_surf)

        """Algebraic equation for Anode electric potential boundary condition"""
        res[offset + ptr['Phi_ed']] = i_el_m - i_el_p + i_io_m - i_io_p

        
# %%
        """================================================================="""
        """======================SEPARATOR INTERIOR========================="""
        offsets = sep.offsets; ptr = sep.ptr
        
        for j in np.arange(1, sep.npoints):
            # Save previous node outlet conditions as new inlet conditions
            i_io_m = i_io_p
            N_elyte_m = N_elyte_p
            s1 = s2
            
            # Set NEXT separator node conditions
            offset = int(offsets[j])
            
            s2 = set_state_sep(offset, SV, elyte, ptr)
            
            # Shift back to THIS node
            offset = int(sep.offsets[j-1])
            
            # Average species and overall molar concentration at the volume 
            #   interface:
            transport.C_k = 0.5*(s2['X_k_el']*s2['rho_el'] 
                           + s1['X_k_el']*s1['rho_el'])
            transport.rho_bar = 0.5*(s2['rho_el'] + s1['rho_el'])

            # Diffusion and migration coefficients:
            D_k, D_k_migr = transport.coeffs()
            
            # Electrolyte species molar fluxes and associated ionic current 
            #   density:
            N_elyte_p, i_io_p = elyte_flux(s1, s2, sep.dyInv, sep, D_k, D_k_migr)
        
            """Change in electrolyte composition"""
            res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
            - (((N_elyte_m - N_elyte_p)*sep.dyInv)/s1['rho_el']/sep.eps_elyte))
            
            """Algebraic equation for electrolyte potential"""
            res[offset + ptr['Phi']] = i_io_m - i_io_p
            
        """=================SEPARATOR-CATHODE BOUNDARY======================="""
        
        i_io_m = i_io_p
        N_elyte_m = N_elyte_p
        s1 = s2
        
        # Read properties from the NEXT node, (i.e. the first cathode node, j=0)
        j = 0; offset = int(ca.offsets[j])
        s2 = set_state(offset, SV, cathode, cathode_s, elyte, conductor, ca.ptr)
        
        # Shift back to THIS node, the final separator node:
        j = sep.npoints-1; offset = int(offsets[j])
        
        # No electronic current:
        i_el_p = 0
        
        # Distance between node centers:
        dyInv_boundary = 1/(0.5*(1/ca.dyInv + 1/sep.dyInv))
        
        # Set conditions at the interface between nodes:
        transport.C_k = (s2['X_k_el']*s2['rho_el'] 
                           + s1['X_k_el']*s1['rho_el'])/2.
        transport.rho_bar = (s2['rho_el'] + s1['rho_el'])/2.
        
        # Diffusion and migration coefficients:
        D_k, D_k_migr = transport.coeffs()

        # Electrolyte species molar fluxes and associated ionic current density:
        N_elyte_p, i_io_p = elyte_flux(s1, s2, dyInv_boundary, sep, D_k, D_k_migr)
                
        """Change in electrolyte composition"""
        res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
        - (((N_elyte_m - N_elyte_p)*sep.dyInv)/s1['rho_el']/sep.eps_elyte))
      
        """Algebraic equation for electrolyte potential"""
        res[offset + ptr['Phi']] = i_io_m - i_io_p

# %%
        """================================================================="""
        """===========================CATHODE==============================="""
        offsets = ca.offsets; ptr = ca.ptr
        
        # Array of cathode "shell" radii, relative to the overall particle 
        #   radius.  This is used to scale the intercalation diffusion fluxes:
        k = np.arange(0, ca.nshells+1)/ca.nshells
                
        """=========================CATHODE============================="""
        """INTERIOR NODES"""
        
        for j in np.arange(1, ca.npoints):
            # Save previous node outlet conditions as new inlet conditions
            N_elyte_m = N_elyte_p
            i_io_m = i_io_p
            i_el_m = i_el_p
            s1 = s2
            
            # Read properties from NEXT node:
            offset = int(offsets[j])
            s2 = set_state(offset, SV, cathode, cathode_s, elyte, conductor, ptr)
            
            # Shift back to THIS node, set THIS node outlet fluxes:
            offset = int(offsets[j-1])

            # Electronic current from Ohm's law:
            i_el_p = ca.sigma_eff_ed*(s1['phi_ed'] - s2['phi_ed'])*ca.dyInv
            
            # Set average species and overall molar concentrations at interface 
            #   between volumes:
            transport.C_k = (s2['X_k_el']*s2['rho_el'] 
                           + s1['X_k_el']*s1['rho_el'])/2.
            transport.rho_bar = (s2['rho_el'] + s1['rho_el'])/2.

            # Diffusion and migration coefficients:
            D_k, D_k_migr = transport.coeffs()
            
            # Electrolyte species molar fluxes and associated ionic current 
            #   density:
            N_elyte_p, i_io_p = elyte_flux(s1, s2, ca.dyInv, ca, D_k, D_k_migr)
            
            # Faradaic current density:s
            i_Far_1 = -s1['sdot'][ptr['iFar']]*F*ca.A_surf/ca.dyInv

            # Double layer current:
            i_dl = -i_Far_1 - i_el_m + i_el_p

            # Convert the double layer current to an equivalent species 
            #   production rate per unit volume:
            R_dl = np.array((0, 0, -i_dl*ca.dyInv/F, 0))
            
            # Array of Li mole fractions as a function of particle radius:
            X_Li = SV[offset + ptr['X_ed']]

            # Initialize array of Li intercalation diffusion flux:
            DiffFlux = np.zeros([ca.nshells+1])

            # Calculate Fickian diffusion flux:
            DiffFlux[1:-1] = ca.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/ca.dr
            # Flux at the surface is equal to the species production rate from 
            #   the faradaic charge transfer reaction:
            DiffFlux[-1] = -s1['sdot'][ca.ptr['iFar']]/s1['rho_ed']

            """Calculate the change in X_LiCoO2 in the particle interior"""
            res[offset + ptr['X_ed']] = (SV_dot[offset + ptr['X_ed']]
            - ((DiffFlux[1:]*k[1:]**2 - DiffFlux[0:-1]*k[0:-1]**2)
            *ca.A_surf/ca.eps_active/ca.eps_ed/ca.V_shell))
            
            """Change in electrolyte composition"""
            res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
            - (((N_elyte_m - N_elyte_p)*ca.dyInv + s1['sdot']*ca.A_surf + R_dl)
            /s1['rho_el']/ca.eps_elyte))
            
            """Double-layer voltage"""
            res[offset + ptr['Phi_dl']] = (SV_dot[offset + ptr['Phi_dl']]
            - (-i_Far_1 + i_io_m - i_io_p)*ca.dyInv/ca.C_dl/ca.A_surf)
            
            """Algebraic equation for CATHODE electric potential"""
            res[offset + ptr['Phi_ed']] = (i_el_m - i_el_p + i_io_m - i_io_p)
        
# %%
        """=========================CATHODE============================="""
        """current collector boundary"""
        N_elyte_m = N_elyte_p
        i_io_m = i_io_p
        i_el_m = i_el_p
        s1 = s2
        
        # FINAL node
        j = ca.npoints-1; offset = int(offsets[j])
        
        # Boundary conditions:
        i_io_p = 0
        N_elyte_p = 0
        i_el_p = i_ext
        
        # Faradaic current density (per unit geometric area)
        i_Far_1 = -s1['sdot'][ptr['iFar']]*F*ca.A_surf/ca.dyInv

        # Double-layer current and associated reaction term:
        i_dl = -i_Far_1 - i_el_m + i_el_p
        R_dl = np.array((0, 0, -i_dl*ca.dyInv/F, 0))
        
        # Array of lithium mole fractions, as a function of cathode particle 
        #   radial position:
        X_Li = SV[offset + ptr['X_ed']]

        # Fickian intercalation diffusion:
        DiffFlux[1:-1] = ca.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/ca.dr
        # Flux at the particle surface equals the produciton rate due to the 
        #   Faradaic charge transfer reaction:
        DiffFlux[-1] = -s1['sdot'][ca.ptr['iFar']]/s1['rho_ed']
                        
        """Calculate the change in X_LiCoO2 in the particle interior"""
        res[offset + ptr['X_ed']] = (SV_dot[offset + ptr['X_ed']]
        - ((DiffFlux[1:]*k[1:]**2 - DiffFlux[0:-1]*k[0:-1]**2)
        *ca.A_surf/ca.eps_active/ca.eps_ed/ca.V_shell))
        
        """Change in electrolyte composition"""
        res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
        - (((N_elyte_m - N_elyte_p)*ca.dyInv + s1['sdot']*ca.A_surf + R_dl)
        /s1['rho_el']/ca.eps_elyte))
        
        """Double-layer voltage"""
        res[offset + ptr['Phi_dl']] = (SV_dot[offset + ptr['Phi_dl']]
        - (-i_Far_1 + i_io_m - i_io_p)*ca.dyInv/ca.C_dl/ca.A_surf)
        
        """Algebraic equation for CATHODE electric potential"""
        # This is used to set the final electric potential, which is that the 
        #   anode current collector is set to the reference value of 0.0 V:
        res[offset + ptr['Phi_ed']] = SV[an.ptr['Phi_ed']]
                        
        return res

# %%
    """====================================================================="""
    """==========================Solver Functions==========================="""
    """====================================================================="""

    # These set termination events for the simulation
    def state_events(self, t, y, yd, sw):
        
        # Initialize anode events
        event1 = np.zeros([an.npoints])
        event2 = np.zeros([an.npoints])
        
        # Terminate if anode double layer potential leaves the bounds (0,5):
        event1 = y[an.ptr_vec['Phi_dl']]
        event2 = 5 - y[an.ptr_vec['Phi_dl']]
            
        # Initialize cathode events
        event5 = np.zeros([ca.npoints])
        event6 = np.zeros([ca.npoints])
        event7 = np.zeros([ca.npoints*ca.nshells])
        event8 = np.zeros([ca.npoints*ca.nshells])
        
        # Terminate if cathode double layer potential is < 0 V:
        event5 = y[ca.ptr_vec['Phi_dl']]

        # Terminate if cathode potential is > 6 V:
        event6 = 6. - y[ca.ptr_vec['Phi_ed']]

        # Terminate if cathode lithium potential leaves the user-supplied 
        #   bounds (X_Li_min, X_Li_max):
        event7 = ca.X_Li_max - y[ca.ptr_vec['X_ed']]
        event8 = y[ca.ptr_vec['X_ed']] - ca.X_Li_min
               
        # Initialize electrolyte events:
        event9  = np.zeros([an.npoints*elyte.n_species])
        event10 = np.zeros([an.npoints*elyte.n_species])
        event11 = np.zeros([ca.npoints*elyte.n_species])
        event12 = np.zeros([ca.npoints*elyte.n_species])

        # Terminate the simulation if any electrolyte species concentrations 
        #   leave the bounds (0,1):
        event9  = 1 - y[an.ptr_vec['X_k_elyte']]
        event10 = y[an.ptr_vec['X_k_elyte']]
        event11 = 1 - y[ca.ptr_vec['X_k_elyte']]
        event12 = y[ca.ptr_vec['X_k_elyte']]

        # Concatenate events into one array
        anode_events = np.array([event1, event2])
        events = np.concatenate((anode_events, event5, event6, 
                                 event7, event8, event9, event10, 
                                 event11, event12))

        return events

    """====================================================================="""

    def handle_event(self, solver, event_info):
        """ This function tells the simulation what to do when an 'event' is encountered (in all cases, terminate the simulation and raise an exception), and provides the appropriate exception message, in each case."""
        state_info = event_info[0] #We are only interested in state events info

        event_ptr = {}
        event_ptr['An_phi1'] = an.npoints
        event_ptr['An_phi2'] = event_ptr['An_phi1'] + an.npoints
        
        event_ptr['ca_phi1'] = event_ptr['An_phi2'] + ca.npoints
        event_ptr['ca_phi2'] = event_ptr['ca_phi1'] + ca.npoints
        event_ptr['ca_Xed1'] = event_ptr['ca_phi2'] + ca.npoints*ca.nshells
        event_ptr['ca_Xed2'] = event_ptr['ca_Xed1'] + ca.npoints*ca.nshells
        
        event_ptr['An_el1'] = event_ptr['ca_Xed2'] + an.npoints*elyte.n_species
        event_ptr['An_el2'] = event_ptr['An_el1'] + an.npoints*elyte.n_species
        event_ptr['ca_el1'] = event_ptr['An_el2'] + ca.npoints*elyte.n_species
        event_ptr['ca_el2'] = event_ptr['ca_el1'] + ca.npoints*elyte.n_species
        
        if any(state_info[0:event_ptr['An_phi1']]):
            print('Cutoff: anode double-layer flipped sign')
            raise TerminateSimulation
        elif any(state_info[event_ptr['An_phi1']:event_ptr['An_phi2']]):
            print('Cutoff: anode double-layer blew up')
            raise TerminateSimulation
        elif any(state_info[event_ptr['An_phi2']:event_ptr['ca_phi1']]):
            print('Cutoff: cahode double layer flipped sign')
            raise TerminateSimulation
        elif any(state_info[event_ptr['ca_phi1']:event_ptr['ca_phi2']]):
            print('Cutoff: Cell potential went over 6 V')
            raise TerminateSimulation
        elif any(state_info[event_ptr['ca_phi2']:event_ptr['ca_Xed1']]):
            print('Cutoff: cahode shell fully lithiated')
            raise TerminateSimulation
        elif any(state_info[event_ptr['ca_Xed1']:event_ptr['ca_Xed2']]):
            print('Cutoff: cahode shell fully de-lithiated')
            raise TerminateSimulation
        elif any(state_info[event_ptr['An_el1']:event_ptr['An_el2']]):
            print('Cutoff: Li+ in electrolyte for anode depleted')
            raise TerminateSimulation
        elif any(state_info[event_ptr['ca_el1']:event_ptr['ca_el2']]):
            print('Cutoff: Li+ in electrolyte for cahode depleted')
            raise TerminateSimulation

    """====================================================================="""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--kinetics')
    parser.add_argument('--H_ca')
    parser.add_argument('--i_ext')
    parser.add_argument('--C_rate')

    args = parser.parse_args()

    SV_eq_df, SV_charge_df, SV_req_df, SV_discharge_df = main(args.kinetics, 
        args.H_ca, args.i_ext, args.C_rate)
