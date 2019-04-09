# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:37:06 2018

@author: dkorff
"""

import numpy as np
import importlib
import cantera as ct
from assimulo.problem import Implicit_Problem

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
        """================================================================="""
        offsets = an.offsets
        
        F = ct.faraday; R = ct.gas_constant; T = Inputs.T
        
        nSV = len(SV)
        res = np.zeros([nSV])
        i_ext = current.get_i_ext()
        print(t, i_ext)
#        print(res)

        """        anode = an.obj['electrode']
        anode_s = an.obj['surf']
        elyte = an.obj['elyte']
        cathode = cat.obj['electrode']
        cathode_s = cat.obj['surf']"""

        nsp_an = anode.n_species; nsp_cat = cathode.n_species

# %%
        """================================================================="""
        """============================ANODE================================"""
        """================================================================="""
        #  --------------------------------
        #  ANODE CURRENT COLLECTOR BOUNDARY
        #  --------------------------------

        # Looking at node 1, j=0, set THIS node conditions
        j = 0; offset = int(offsets[j]); ptr = an.ptr

        N_io_p = 0
        i_io_p = 0
        i_el_p = i_ext
        
        phi_1 = {}
        
        # Shift forward to node 2, j=1, to set NEXT node conditions
        sdot_2, phi_2, X_an_2, rho_k_elyte_2 = \
        Extended_Problem.set_state(offset, SV, anode, anode_s, elyte, ptr)
        
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
            X_an_1 = X_an_2
            rho_k_elyte_1 = rho_k_elyte_2
            phi_1['ed'] = phi_2['ed']
            phi_1['el'] = phi_2['el']
            sdot_1 = sdot_2

            # Shift forward to NEXT node
            offset = int(offsets[j])

            sdot_2, phi_2, X_an_2, rho_k_elyte_2 = \
            Extended_Problem.set_state(offset, SV, anode, anode_s, elyte, ptr)

            # Shift back to THIS node, set THIS node outlet conditions
            offset = int(offsets[j - 1])

            i_el_p = an.sigma_eff_ed*(phi_1['ed'] - phi_2['ed'])*an.dyInv

            N_io_p = (-an.u_Li_elyte*rho_k_elyte_1
                      *(R*T*(rho_k_elyte_2 - rho_k_elyte_1)
                      + Inputs.z_k_elyte*F*(phi_2['el'] - phi_1['el']))*an.dyInv)
            
#                    (-an.u_Li_elyte*elyte.density_mole
#                      *(R*T*(rho_k_elyte_2 - rho_k_elyte_1)
#                      + F*(phi_2['el'] - phi_1['el']))*an.dyInv)

            i_io_p = np.dot(N_io_p,Inputs.z_k_elyte)*F

            i_Far_1 = sdot_1[ptr['iFar']]*F*an.A_surf/an.dyInv
#            print(i_Far_1, sdot_1)

            X_Li = SV[offset + ptr['X_ed']]
            DiffFlux = np.zeros([an.nshells+1])
            DiffFlux[1:-1] = an.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/an.dr
            DiffFlux[-1] = sdot_1[an.ptr['iFar']]/anode.density_mole
#            print(DiffFlux)

            """Calculate the change in X_LiC6 in the particle interior."""
            res[offset + an.ptr['X_ed']] = (SV_dot[offset + ptr['X_ed']]
            - ((DiffFlux[1:]*k[1:]**2 - DiffFlux[0:-1]*k[0:-1]**2)
            * an.A_surf/an.eps_ed/an.V_shell))

            """Change in electrolyte_composition"""
            res[offset + ptr['rho_k_elyte']] = (SV_dot[offset + ptr['rho_k_elyte']]
            - (((N_io_m - N_io_p)*an.dyInv + sdot_1[nsp_an]*an.A_surf)
            /elyte.density_mole/an.eps_elyte))

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
        X_an_1 = X_an_2
        rho_k_elyte_1 = rho_k_elyte_2
        phi_1['ed'] = phi_2['ed']
        phi_1['el'] = phi_2['el']
        sdot_1 = sdot_2

        # Shift forward to NEXT node (first separator node)

        # Shift back to THIS node, set THIS node outlet conditions
        i_el_p = 0

        # Set j to final ANODE node
        j = an.npoints-1; offset = int(an.offsets[j])

        i_Far_1 = sdot_1[ptr['iFar']]*F*an.A_surf/an.dyInv

        i_io_p = i_ext
        #THIS IS TEMPORARY, NON-GENERALIZED CODE:
        N_io_p = np.zeros_like(N_io_p)
        N_io_p[2] = i_io_p/F

        X_Li = SV[offset + an.ptr['X_ed']]
        DiffFlux = np.zeros([an.nshells+1])
        DiffFlux[1:-1] = an.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/an.dr
        DiffFlux[-1] = sdot_1[an.ptr['iFar']]/anode.density_mole

        """Calculate the change in X_LiC6 in the particle interior."""
        res[offset + ptr['X_ed']] = (SV_dot[offset + ptr['X_ed']]
        - ((DiffFlux[1:]*k[1:]**2 - DiffFlux[0:-1]*k[0:-1]**2)
        * an.A_surf/an.eps_ed/an.V_shell))

        """Change in electrolyte_composition"""
        res[offset + ptr['rho_k_elyte']] = (SV_dot[offset + ptr['rho_k_elyte']]
        - (((N_io_m - N_io_p)*an.dyInv + sdot_1[nsp_an]*an.A_surf)
        /elyte.density_mole/an.eps_elyte))

        """Double-layer voltage"""
        res[offset + ptr['Phi_dl']] = (SV_dot[offset + ptr['Phi_dl']]
        - (-i_Far_1 + i_io_m - i_io_p)*an.dyInv/an.C_dl/an.A_surf)

        """Algebraic equation for ANODE electric potential boundary condition"""
        res[offset + ptr['Phi_ed']] = SV[ptr['Phi_ed']]
# %%
        """================================================================="""
        """==========================SEPARATOR=============================="""
        """================================================================="""

# %%
        # Looking at LAST node in separator

# %%
        """================================================================="""
        """===========================CATHODE==============================="""
        """================================================================="""



# %%
        """=========================CATHODE============================="""
        """current collector boundary"""

#        print(res)
        return res

    """====================================================================="""
    """===========================Model Functions==========================="""
    """====================================================================="""
    
    def set_state(offset, SV, ed, surf, el, ptr):
        X_ed = SV[offset + ptr['X_ed'][-1]]
        rho_k_el = SV[offset + ptr['rho_k_elyte']]
        
        phi_elec_ed = SV[offset + ptr['Phi_ed']]
        phi_elec_el = SV[offset + ptr['Phi_dl']] + phi_elec_ed
        
        ed.X = [X_ed, 1 - X_ed]
        ed.electric_potential = phi_elec_ed
        
        elyte.Y = rho_k_el/np.sum(rho_k_el)
        elyte.electric_potential = phi_elec_el
        
        sdot = surf.net_production_rates
        phi = {}
        phi['ed'] = phi_elec_ed
        phi['el'] = phi_elec_el
        
        return sdot, phi, X_ed, rho_k_el
    
    """====================================================================="""
    """==========================Solver Functions==========================="""
    """====================================================================="""
# %%
    def state_events(self, t, y, yd, sw):
        event1 = np.zeros([an.npoints])
        event2 = np.zeros([an.npoints])
        event3 = np.zeros([an.nshells])
        event4 = np.zeros([an.nshells])

        for j in np.arange(0, an.npoints):
            offset = j*an.nVars

            event1[j] = (y[offset + an.ptr['Phi_dl']])
#            event2[j] = ( - y[offset + an.ptr['Phi_dl']])

            for i in np.arange(0, an.nshells):
                event3[i] = an.X_Li_max - y[offset + an.ptr['X_ed'][i]]
                event4[i] = y[offset + an.ptr['X_ed'][i]] - an.X_Li_min

#        event5 = np.zeros([cat.npoints])
#        event6 = np.zeros([cat.npoints])
#        event7 = np.zeros([cat.nshells])
#        event8 = np.zeros([cat.nshells])
#
#        for j in np.arange(0, cat.npoints):
#            offset = j*cat.nVars + an.npoints*an.nVars + sep.npoints*sep.nVars
#
#            event5[j] = (y[offset + cat.ptr['Phi_dl']])
#            event6[j] = (y[offset + cat.ptr['Phi_dl']] - 5)
#
#            for i in np.arange(0, cat.nshells):
#                event7[i] = y[offset + cat.ptr['X_ed'][i]] - (1 - cat.X_Li_max)
#                event8[i] = (1 - cat.X_Li_min) - y[offset + cat.ptr['X_ed'][i]]
#
#        event9 = np.zeros([sep.npoints])
#        event10 = np.zeros([sep.npoints])
#        for j in np.arange(0, sep.npoints):
#            offset = an.npoints*an.nVars
#            event9[j] = 1 - y[offset + sep.ptr['X_elyte']]
#            event10[j] = y[offset + sep.ptr['X_elyte']]

        events = np.concatenate((event1, event2, event3, event4))

        return events

    """====================================================================="""

    def handle_event(self, solver, event_info):
        while True:
            self.event_switch(solver, event_info)
            self.init_mode(solver)

            if not True in event_info:
                break

    def event_switch(self, solver, event_info):
        if not all(event_info):
            solver.sw = [not solver.sw]

    def init_mode(self, solver):
        an.t_flag = solver.t
        if current.get_i_ext() != 0:
            current.set_i_ext(0)
            
