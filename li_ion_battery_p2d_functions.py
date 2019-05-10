# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:37:06 2018

@author: dkorff
"""

import numpy as np
import importlib
import cantera as ct
from assimulo.problem import Implicit_Problem
from assimulo.exception import TerminateSimulation

import li_ion_battery_p2d_init
importlib.reload(li_ion_battery_p2d_init)
from li_ion_battery_p2d_init import anode as an
from li_ion_battery_p2d_init import cathode as cat
from li_ion_battery_p2d_init import separator as sep
from li_ion_battery_p2d_init import Inputs
from li_ion_battery_p2d_init import anode_obj as anode
from li_ion_battery_p2d_init import anode_surf_obj as anode_s
from li_ion_battery_p2d_init import elyte_obj as elyte
from li_ion_battery_p2d_init import cathode_surf_obj as cathode_s
from li_ion_battery_p2d_init import cathode_obj as cathode
from li_ion_battery_p2d_init import conductor_obj as conductor
from li_ion_battery_p2d_init import solver_inputs
from li_ion_battery_p2d_init import current


class Extended_Problem(Implicit_Problem):
    def Battery_Func(t, SV, SV_dot):
        """================================================================="""
        """==========================INITIALIZE============================="""
        offsets = an.offsets; F = ct.faraday; R = ct.gas_constant; T = Inputs.T
        
        nSV = len(SV)
        res = np.zeros([nSV])
        i_ext = current.get_i_ext()

# %%
        """================================================================="""
        """============================ANODE================================"""
        # Looking at node 0, CC boundary "outlet" conditions
        j = 0; offset = int(offsets[j]); ptr = an.ptr

        N_io_p = 0; i_io_p = 0; i_el_p = i_ext
        
        phi_1 = {}
        phi_2 = {}
                
        # Shift forward to node 1, j=0, to set FIRST node conditions
        sdot_2, phi_2, X_k_el_2, rho_el_2, rho_ed_2 = \
            Extended_Problem.set_state(offset, SV, anode, anode_s, elyte, 
                                       conductor, ptr)
        
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
            X_k_el_1 = X_k_el_2
            phi_1['ed'] = phi_2['ed']
            phi_1['el'] = phi_2['el']
            sdot_1 = sdot_2
            rho_el_1 = rho_el_2
            rho_ed_1 = rho_ed_2

            # Shift forward to NEXT node
            offset = int(offsets[j])

            sdot_2, phi_2, X_k_el_2, rho_el_2, rho_ed_2 = \
            Extended_Problem.set_state(offset, SV, anode, anode_s, elyte, 
                                       conductor, ptr)

            # Shift back to THIS node, set THIS node outlet conditions
            offset = int(offsets[j - 1])

            i_el_p = an.sigma_eff_ed*(phi_1['ed'] - phi_2['ed'])*an.dyInv
            
            C_k = (X_k_el_2*rho_el_2 + X_k_el_1*rho_el_1)/2.
            C_0 = (rho_el_2 + rho_el_1)/2.
            X_k = (X_k_el_2 + X_k_el_1)/2.

            N_io_p = (-an.u_Li_elyte*(R*T*C_0*(X_k_el_2 - X_k_el_1)
                      + Inputs.z_k_elyte*F*C_k*(phi_2['el'] - phi_1['el']))*an.dyInv)
            
            i_io_p = np.dot(N_io_p,Inputs.z_k_elyte)*F

            i_Far_1 = -sdot_1[ptr['iFar']]*F*an.A_surf/an.dyInv

            X_Li = SV[offset + ptr['X_ed']]
            DiffFlux = np.zeros([an.nshells+1])
            DiffFlux[1:-1] = an.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/an.dr
            DiffFlux[-1] = -sdot_1[an.ptr['iFar']]/rho_ed_1

            """Calculate the change in X_LiC6 in the particle interior."""
            res[offset + an.ptr['X_ed']] = (SV_dot[offset + ptr['X_ed']]
            - ((DiffFlux[1:]*k[1:]**2 - DiffFlux[0:-1]*k[0:-1]**2)
            * an.A_surf/an.eps_ed/an.V_shell))

            """Change in electrolyte_composition"""
            res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
            - (((N_io_m - N_io_p)*an.dyInv + sdot_1*an.A_surf)
            /rho_el_1/an.eps_elyte))

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
        X_k_el_1 = X_k_el_2
        phi_1['ed'] = phi_2['ed']
        phi_1['el'] = phi_2['el']
        sdot_1 = sdot_2
        rho_el_1 = rho_el_2
        rho_ed_1 = rho_ed_2
#        
        # Shift forward to NEXT node, first separator node (j=0)
        j = 0; offset = int(sep.offsets[j])
        
        phi_2['el'] = SV[offset + sep.ptr['Phi']]
        X_k_el_2 = SV[offset + sep.ptr['X_k_elyte']]
        
        elyte.X = X_k_el_2
        elyte.electric_potential = phi_2['el']
        rho_el_2 = elyte.density_mole

        # Shift back to THIS node, set THIS node outlet conditions
        i_el_p = 0

        # Set j to final ANODE node
        j = an.npoints-1; offset = int(an.offsets[j])

        i_Far_1 = -sdot_1[ptr['iFar']]*F*an.A_surf/an.dyInv
        
#        i_io_p = i_ext
#        N_io_p = np.zeros([elyte.n_species])
#        N_io_p[2] = i_io_p/F

        C_k = (X_k_el_2*rho_el_2+ X_k_el_1*rho_el_1)/2.
        C_0 = (rho_el_2 + rho_el_1)/2.
        X_k = (X_k_el_2 + X_k_el_1)/2.
        
        dyInv_boundary = 1/(0.5*(1/an.dyInv + 1/sep.dyInv))
            
        N_io_p = (-an.u_Li_elyte*(R*T*C_0*(X_k_el_2 - X_k_el_1)
                  + Inputs.z_k_elyte*F*C_k*(phi_2['el'] - phi_1['el']))*dyInv_boundary)

        i_io_p = np.dot(N_io_p,Inputs.z_k_elyte)*F
        
        X_Li = SV[offset + an.ptr['X_ed']]
        DiffFlux = np.zeros([an.nshells+1])
        DiffFlux[1:-1] = an.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/an.dr
        DiffFlux[-1] = -sdot_1[an.ptr['iFar']]/rho_ed_1
    
        """Calculate the change in X_LiC6 in the particle interior."""
        res[offset + ptr['X_ed']] = (SV_dot[offset + ptr['X_ed']]
        - ((DiffFlux[1:]*k[1:]**2 - DiffFlux[0:-1]*k[0:-1]**2)
        * an.A_surf/an.eps_ed/an.V_shell))

        """Change in electrolyte_composition"""
        res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
        - (((N_io_m - N_io_p)*an.dyInv + sdot_1*an.A_surf)
        /rho_el_1/an.eps_elyte))

        """Double-layer voltage"""
        res[offset + ptr['Phi_dl']] = (SV_dot[offset + ptr['Phi_dl']]
        - (-i_Far_1 + i_io_m - i_io_p)*an.dyInv/an.C_dl/an.A_surf)

        """Algebraic equation for ANODE electric potential boundary condition"""
        res[offset + ptr['Phi_ed']] = SV[an.ptr['Phi_ed']]
#        (i_el_m - i_el_p + i_io_m - i_io_p)
#        SV[an.ptr['Phi_ed']]
        
# %%
        """================================================================="""
        """==========================SEPARATOR=============================="""
        offsets = sep.offsets; ptr = sep.ptr
        
        for j in np.arange(1, sep.npoints):
            # Save previous node outlet conditions as new inlet conditions
            i_io_m = i_io_p
            N_io_m = N_io_p
            X_k_el_1 = X_k_el_2
            rho_el_1 = rho_el_2
            phi_1['el'] = phi_2['el']
            
            # Set NEXT separator node conditions
            offset = int(offsets[j])
            
            phi_2['el'] = SV[offset + ptr['Phi']]
            X_k_el_2 = SV[offset + sep.ptr['X_k_elyte']]
        
            elyte.X = X_k_el_2
            elyte.electric_potential = phi_2['el']
            rho_el_2 = elyte.density_mole
            
            # Shift back to THIS node
            offset = int(sep.offsets[j-1])
            
            C_k = (X_k_el_2*rho_el_2+ X_k_el_1*rho_el_1)/2.
            C_0 = (rho_el_2 + rho_el_1)/2.
            X_k = (X_k_el_2 + X_k_el_1)/2.
            
            N_io_p = (-an.u_Li_elyte*(R*T*C_0*(X_k_el_2 - X_k_el_1)
                      + Inputs.z_k_elyte*F*C_k*(phi_2['el'] - phi_1['el']))*an.dyInv)

            i_io_p = np.dot(N_io_p,Inputs.z_k_elyte)*F
        
            """Change in electrolyte composition"""
            res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
            - (((N_io_m - N_io_p)*sep.dyInv)/rho_el_1/sep.eps_elyte))
            
            """Algebraic equation for electrolyte potential"""
            res[offset + ptr['Phi']] = i_io_m - i_io_p
            
        """==========================SEPARATOR=============================="""
        """Cathode boundary"""
        
        i_io_m = i_io_p
        N_io_m = N_io_p
        X_k_el_1 = X_k_el_2
        rho_el_1 = rho_el_2
        phi_1['el'] = phi_2['el']
        
        # Shift forward to NEXT node, first cathode node (j=0)
        j = 0; offset = int(cat.offsets[j])
        sdot_2, phi_2, X_k_el_2, rho_el_2, rho_ed_2 = \
            Extended_Problem.set_state(offset, SV, cathode, cathode_s, elyte, 
                                       conductor, cat.ptr)
        
        # Shift to final separator node
        j = sep.npoints-1; offset = int(offsets[j])
        
        i_el_p = 0
        
        C_k = (X_k_el_2*rho_el_2+ X_k_el_1*rho_el_1)/2.
        C_0 = (rho_el_2 + rho_el_1)/2.
        X_k = (X_k_el_2 + X_k_el_1)/2.
        
        dyInv_boundary = 1/(0.5*(1/cat.dyInv + 1/sep.dyInv))
            
        N_io_p = (-cat.u_Li_elyte*(R*T*C_0*(X_k_el_2 - X_k_el_1)
                  + Inputs.z_k_elyte*F*C_k*(phi_2['el'] - phi_1['el']))*dyInv_boundary)

        i_io_p = np.dot(N_io_p,Inputs.z_k_elyte)*F
        
#        i_io_p = i_ext
#        N_io_p = np.zeros([elyte.n_species])
#        N_io_p[2] = i_io_p/F
                
        """Change in electrolyte composition"""
        res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
        - (((N_io_m - N_io_p)*sep.dyInv)/rho_el_1/sep.eps_elyte))
      
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
            X_k_el_1 = X_k_el_2
            phi_1['ed'] = phi_2['ed']
            phi_1['el'] = phi_2['el']
            sdot_1 = sdot_2
            rho_el_1 = rho_el_2
            rho_ed_1 = rho_ed_2
            
            # Shift forward to NEXT node
            offset = int(offsets[j])
            
            sdot_2, phi_2, X_k_el_2, rho_el_2, rho_ed_2 = \
            Extended_Problem.set_state(offset, SV, cathode, cathode_s, elyte, 
                                       conductor, ptr)
            
            # Shift back to THIS node, set THIS node outlet conditions
            offset = int(offsets[j-1])
            
            i_el_p = cat.sigma_eff_ed*(phi_1['ed'] - phi_2['ed'])*cat.dyInv
            
            C_k = (X_k_el_2*rho_el_2+ X_k_el_1*rho_el_1)/2.
            C_0 = (rho_el_2 + rho_el_1)/2.
            X_k = (X_k_el_2 + X_k_el_1)/2.
            
            N_io_p = (-cat.u_Li_elyte*(R*T*C_0*(X_k_el_2 - X_k_el_1)/X_k
                      + Inputs.z_k_elyte*F*C_k*(phi_2['el'] - phi_1['el']))*cat.dyInv)
            
            i_io_p = np.dot(N_io_p, Inputs.z_k_elyte)*F
            
            i_Far_1 = -sdot_1[ptr['iFar']]*F*cat.A_surf/cat.dyInv
            
            X_Li = SV[offset + ptr['X_ed']]
            DiffFlux = np.zeros([cat.nshells+1])
            DiffFlux[1:-1] = cat.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/cat.dr
            DiffFlux[-1] = -sdot_1[ptr['iFar']]/rho_ed_1

            """Calculate the change in X_LiCoO2 in the particle interior"""
            res[offset + ptr['X_ed']] = (SV_dot[offset + ptr['X_ed']]
            - ((DiffFlux[1:]*k[1:]**2 - DiffFlux[0:-1]*k[0:-1]**2)
            *cat.A_surf/cat.eps_ed/cat.V_shell))
            
            """Change in electrolyte composition"""
            res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
            - (((N_io_m - N_io_p)*cat.dyInv + sdot_1*cat.A_surf)
            /rho_el_1/cat.eps_elyte))
            
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
        X_k_el_1 = X_k_el_2
        phi_1['ed'] = phi_2['ed']
        phi_1['el'] = phi_2['el']
        sdot_1 = sdot_2
        rho_ed_1 = rho_ed_2
        
        # FINAL node
        j = cat.npoints-1; offset = int(offsets[j])
        
        i_io_p = 0
        N_io_p = 0
        i_el_p = i_ext
        
        i_Far_1 = -sdot_1[ptr['iFar']]*F*cat.A_surf/cat.dyInv
        
        X_Li = SV[offset + ptr['X_ed']]
        DiffFlux = np.zeros([cat.nshells+1])
        DiffFlux[1:-1] = cat.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/cat.dr
        DiffFlux[-1] = -sdot_1[ptr['iFar']]/rho_ed_1
                        
        """Calculate the change in X_LiCoO2 in the particle interior"""
        res[offset + ptr['X_ed']] = (SV_dot[offset + ptr['X_ed']]
        - ((DiffFlux[1:]*k[1:]**2 - DiffFlux[0:-1]*k[0:-1]**2)
        *cat.A_surf/cat.eps_ed/cat.V_shell))
        
        """Change in electrolyte composition"""
        res[offset + ptr['X_k_elyte']] = (SV_dot[offset + ptr['X_k_elyte']]
        - (((N_io_m - N_io_p)*cat.dyInv + sdot_1*cat.A_surf)
        /rho_el_1/cat.eps_elyte))
        
        """Double-layer voltage"""
        res[offset + ptr['Phi_dl']] = (SV_dot[offset + ptr['Phi_dl']]
        - (-i_Far_1 + i_io_m - i_io_p)*cat.dyInv/cat.C_dl/cat.A_surf)
        
        """Algebraic equation for CATHODE electric potential"""
        res[offset + ptr['Phi_ed']] = (i_el_m - i_el_p + i_io_m - i_io_p)
                
        return res

    """====================================================================="""
    """===========================Model Functions==========================="""
    """====================================================================="""
    
    def set_state(offset, SV, ed, surf, el, conductor, ptr):
        X_ed = SV[offset + ptr['X_ed'][-1]]
        X_k_elyte = SV[offset + ptr['X_k_elyte']]
        
        phi_elec_ed = SV[offset + ptr['Phi_ed']]
        phi_elec_el = SV[offset + ptr['Phi_dl']] + phi_elec_ed
        
        ed.X = [X_ed, 1 - X_ed]
        ed.electric_potential = phi_elec_ed
        rho_ed = ed.density_mole
        
        conductor.electric_potential = phi_elec_ed
        
        elyte.X = X_k_elyte
        elyte.electric_potential = phi_elec_el
        rho_el = elyte.density_mole
        
        sdot = surf.get_net_production_rates(el)
        phi = {}
        phi['ed'] = phi_elec_ed
        phi['el'] = phi_elec_el
        
        return sdot, phi, X_k_elyte, rho_el, rho_ed

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
               
        # Separator events
        event9 = np.zeros([sep.npoints])
        event10 = np.zeros([sep.npoints])
        
#        event9 = 1 - y[sep.ptr_vec['X_k_elyte'][an.ptr['iFar']]]
#        
#        for j in np.arange(0, sep.npoints):
#            offset = an.npoints*an.nVars
#            event9[j] = 0.995 - y[offset + sep.ptr['X_k_elyte'][an.ptr['iFar']]]
#            event10[j] = y[offset + sep.ptr['X_k_elyte'][an.ptr['iFar']]] -0.005

        # Concatenate events into one array
        events = np.concatenate((event1, event2, event3, event4, 
                                 event5, event6, event7, event8, 
                                 event9, event10))

        return events

    """====================================================================="""

    def handle_event(self, solver, event_info):
        
        state_info = event_info[0] #We are only interested in state events info

        if not all(state_info):
            an.set_tflag(solver.t)
            print('Cutoff')
            raise TerminateSimulation

    """====================================================================="""
            
