# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:59:27 2018

@author: dkorff
"""

import importlib

import li_ion_battery_p2d_inputs
importlib.reload(li_ion_battery_p2d_inputs)
from li_ion_battery_p2d_inputs import Inputs

from li_ion_battery_p2d_init import initialize as init
an, ca, sep, solver_inputs, current, battery, anode, anode_s, elyte_obj, cathode, cathode_s, conductor = init()

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plot_potential(V_an, V_ca, SV, stage, yax, fig, axes):
    
    if stage == 'Discharging':
        showlegend = 1
    else:
        showlegend = 0
    fontsize = 26
    
    SV_df = SV.copy()
    if Inputs.grav_cap_method == 'cathode':
        SV_df.loc[:, 'Time'] *= -current.i_ext_amp/3600/ca.rho_ed/ca.H/ca.eps_ed
    elif Inputs.grav_cap_method == 'cell':
        SV_df.loc[:, 'Time'] *= -current.i_ext_amp/3600/battery.rho/battery.H/battery.eps
    t = SV_df['Time']
    
    if Inputs.flag_cathode:
        index = []
        for i in np.arange(0, ca.npoints):
            index_add = [2*i]
            index = np.append(index, index_add)
            
        V = [V_ca[i] for i in index.astype(int)]
#        SOC = SV_df.filter(like="X_ca").sum(1)/(ca.npoints*ca.nshells)
        
    elif not Inputs.flag_cathode:
        index = []
        for i in np.arange(0, an.npoints):
            index_add = [1 + 2*i]
            index = np.append(index, index_add)
            
        V = [V_an[i] for i in index.astype(int)]
            
    #    line_style = ['v-', 'o-', '^-', 's-', 'h-', '+-']
        
    # Plot cathode and double-layer potential
    SV_plot = SV_df.plot(x='Time', y=V, ax=axes[yax], xlim=[0,t.iloc[-1]])
    SV_plot.set_title(stage, fontsize = fontsize)
    SV_plot.set_ylabel('Voltages [V]', fontsize = fontsize)
    SV_plot.set_xlabel('Capacity [Ah/kg]', fontsize = fontsize).set_visible(True)
    SV_plot.legend(loc=2, bbox_to_anchor=(1.05, 1), ncol=1, borderaxespad=0,
                   frameon=False).set_visible(showlegend)
    SV_plot.tick_params(axis='both', labelsize=18)
    SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    SV_plot.set_ylim((0,5.))

"""========================================================================="""

def plot_electrode(X_ca, SV, stage, yax, fig, axes):
    
    SV_df = SV.copy()
    if Inputs.grav_cap_method == 'cathode':
        SV_df.loc[:, 'Time'] *= -current.i_ext_amp/3600/ca.rho_ed/ca.H/ca.eps_ed
    elif Inputs.grav_cap_method == 'cell':
        SV_df.loc[:, 'Time'] *= -current.i_ext_amp/3600/battery.rho/battery.H/battery.eps
    t = SV_df['Time']
    fontsize = 26
    
    if stage == 'Discharging':
        showlegend = 1
    else:
        showlegend = 0
    
    if Inputs.flag_anode and Inputs.flag_cathode:
        axes_ca = axes[yax]
    elif Inputs.flag_anode and not Inputs.flag_cathode:
        axes_an = axes[yax]
    elif Inputs.flag_cathode and not Inputs.flag_anode:
        axes_ca = axes[yax]
    
    if Inputs.flag_cathode:
        index = []
        for i in np.arange(0, ca.npoints):
            offset = i*ca.nshells
            index_add = [0+offset, ca.nshells-1+offset]
            index = np.append(index, index_add)
        
        X_ca_1 = [X_ca[i] for i in index.astype(int)]
    
        # Plot cathode composition
        SV_plot = SV_df.plot(x = 'Time', y = X_ca_1, ax = axes_ca, xlim = [0, t.iloc[-1]],
                             ylim = [-0.1, 1.1])
        SV_plot.set_title(stage, fontsize = fontsize).set_visible(False)
        SV_plot.set_ylabel('$X_{LiCoO2}$', fontsize = fontsize)
        SV_plot.set_xlabel('Capacity [Ah/kg]', fontsize = fontsize).set_visible(True)
        SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, 
                       borderaxespad = 0, frameon = False).set_visible(showlegend)
        SV_plot.tick_params(axis='both', labelsize = 18)
        SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    return

"""========================================================================="""

def plot_elyte(rho_k_an, rho_k_ca, rho_k_sep, SV, stage, yax, fig, axes):
    
    SV_df = SV.copy()
    if Inputs.grav_cap_method == 'cathode':
        SV_df.loc[:, 'Time'] *= -current.i_ext_amp/3600/ca.rho_ed/ca.H/ca.eps_ed
    elif Inputs.grav_cap_method == 'cell':
        SV_df.loc[:, 'Time'] *= -current.i_ext_amp/3600/battery.rho/battery.H/battery.eps
    t = SV_df['Time']
    fontsize = 22
    
    if stage == 'Discharging':
        showlegend = 1
    else:
        showlegend = 0
        
    axes_an = axes[0, yax]
    axes_sep = axes[Inputs.flag_sep, yax]
    axes_ca = axes[Inputs.flag_sep+Inputs.flag_cathode, yax]
    
    if Inputs.flag_anode:
        index = []
        for i in np.arange(0, an.npoints):
            offset = i*elyte_obj.n_species
            index_add = [2+offset]
            index = np.append(index, index_add)
            
        rho_Li_an = [rho_k_an[i] for i in index.astype(int)]
    
    if Inputs.flag_cathode:
        index = []
        for i in np.arange(0, ca.npoints):
            offset = i*elyte_obj.n_species
            index_add = [2+offset]
            index = np.append(index, index_add)
            
        rho_Li_ca = [rho_k_ca[i] for i in index.astype(int)]
        
    if Inputs.flag_sep:
        index = []
        for i in np.arange(0, sep.npoints):
            offset = i*elyte_obj.n_species
            index_add = [2+offset]
            index = np.append(index, index_add)
            
        rho_Li_sep = [rho_k_sep[i] for i in index.astype(int)]
    
    if Inputs.flag_anode and Inputs.flag_cathode:
        max_val = max(max(SV_df[rho_Li_an].max()), max(SV_df[rho_Li_ca].max()))
        ymax = max_val + max_val/10
        min_val = min(min(SV_df[rho_Li_an].min()), min(SV_df[rho_Li_ca].min()))
        ymin = min_val - min_val/10
        
        if yax > 0:
            y_old = min(axes[0, yax-1].get_ylim(), axes[1, yax-1].get_ylim())
            if y_old[0] < ymin:
                ymin = y_old[0]
            if y_old[1] > ymax:
                ymax = y_old[1]
        
    if Inputs.flag_anode:
        # Plot elyte composition in anode
        SV_plot = SV_df.plot(x = 'Time', y = rho_Li_an, ax = axes_an, 
                             xlim = [0, t.iloc[-1]], ylim = [ymin, ymax])
        SV_plot.set_title(stage, fontsize = fontsize)
        SV_plot.set_ylabel(r'$X_{Li^+, an}$', fontsize = fontsize)
        SV_plot.set_xlabel('Capacity [Ah/kg]', fontsize = fontsize).set_visible(False)
        SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, borderaxespad = 0,
                       frameon = False).set_visible(showlegend)
        SV_plot.tick_params(axis='both', labelsize = 18)
        SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    if Inputs.flag_cathode:
        # Plot elyte composition in cathode
        SV_plot = SV_df.plot(x = 'Time', y = rho_Li_ca, ax = axes_ca, 
                             xlim = [0, t.iloc[-1]], ylim = [ymin, ymax])
        SV_plot.set_title(stage, fontsize = fontsize).set_visible(False)
        SV_plot.set_ylabel(r'$X_{Li^+, ca}$', fontsize = fontsize)
        SV_plot.set_xlabel('Capacity [Ah/kg]', fontsize = fontsize).set_visible(False)
        SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, borderaxespad = 0,
                       frameon = False).set_visible(showlegend)
        SV_plot.tick_params(axis='both', labelsize = 18)
        SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
    if Inputs.flag_sep:
        # Plot elyte composition in separator
        SV_plot = SV_df.plot(x = 'Time', y = rho_Li_sep, ax = axes_sep, 
                             xlim = [0, t.iloc[-1]], ylim = [ymin, ymax])
        SV_plot.set_title(stage, fontsize = fontsize).set_visible(False)
        SV_plot.set_ylabel(r'$X_{Li^+, sep}$', fontsize = fontsize)
        SV_plot.set_xlabel('Capacity [Ah/kg]', fontsize = fontsize).set_visible(True)
        SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, borderaxespad = 0,
                       frameon = False).set_visible(showlegend)
        SV_plot.tick_params(axis='both', labelsize = 18)
        SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    return

"""========================================================================="""

def plot_cap(SV_ch_df, SV_dch_df, rate_tag, i_ext, flag_plot, tags):
    fontsize = 18
    
    V_ch = np.array(SV_ch_df[tags['Phi_ca'][-2]])
    V_dch = np.array(SV_dch_df[tags['Phi_ca'][-2]])
    t_ch = np.array(SV_ch_df['Time'])
    t_dch = np.array(SV_dch_df['Time'])
    dt_ch = t_ch - t_ch[0]
    dt_dch = t_dch - t_dch[0]
    
    # Plot charge-discharge curve
    if Inputs.cap_method == 'areal':
        Capacity_charge = -dt_ch*i_ext/3600         # A-h/m^2
        Capacity_discharge = -dt_dch*i_ext/3600   # A-h/m^2
    elif Inputs.cap_method == 'grav' and Inputs.grav_cap_method == 'cathode':
        Capacity_charge = -dt_ch*i_ext/3600/ca.rho_ed/ca.H/ca.eps_ed
        Capacity_discharge = -dt_dch*i_ext/3600/ca.rho_ed/ca.H/ca.eps_ed
    elif Inputs.cap_method == 'grav' and Inputs.grav_cap_method == 'cell':
        Capacity_charge = -dt_ch*i_ext/3600/battery.mass#battery.rho/battery.H/battery.eps
        Capacity_discharge = -dt_dch*i_ext/3600/battery.mass#battery.rho/battery.H/battery.eps
    
    SV_ch = SV_ch_df.copy()
    if Inputs.grav_cap_method == 'cathode':
        SV_ch.loc[:, 'Time'] *= -current.i_ext_amp/3600/ca.rho_ed/ca.H/ca.eps_ed
    elif Inputs.grav_cap_method == 'cell':
        SV_ch.loc[:, 'Time'] *= -current.i_ext_amp/3600/battery.mass#rho/battery.H/battery.eps
    SV_dch = SV_dch_df.copy()
    if Inputs.grav_cap_method == 'cathode':
        SV_dch.loc[:, 'Time'] *= -current.i_ext_amp/3600/ca.mass#rho_ed/ca.H/ca.eps_ed
    elif Inputs.grav_cap_method == 'cell':
        SV_dch.loc[:, 'Time'] *= -current.i_ext_amp/3600/battery.mass#rho/battery.H/battery.eps
    
    if flag_plot:
        plt.figure(4, figsize = (8, 6))
        plt.plot(Capacity_charge, V_ch, 'g-')
        plt.plot(Capacity_discharge, V_dch, 'g--')
        plt.title('Potential vs. Capacity', fontsize = fontsize)
        if Inputs.cap_method == 'areal':
            plt.xlabel('$Capacity [Ah/m^2]$', fontsize = fontsize)
        elif Inputs.cap_method == 'grav':
            plt.xlabel('$Capacity [Ah/kg]$', fontsize = fontsize)
        plt.ylabel('Voltage [V]', fontsize = fontsize)
        plt.legend(('Charge', 'Discharge'), loc = 3, fontsize = 14)
        plt.ylim((0,4.5))
        plt.savefig(Inputs.save_path+Inputs.save_name+'Charge_Discharge.png',
            dpi=350)
        plt.show()
    
    Capacity_ch_areal = -dt_ch*i_ext/3600         # A-h/m^2
    Capacity_dch_areal = -dt_dch*i_ext/3600   # A-h/m^2
    Cap_recovered = round(Capacity_dch_areal[-1], 2)
    Cap_stored = round(Capacity_ch_areal[-1], 2)
    Eta_c = round(100*Cap_recovered/Cap_stored, 1)
    
    print('Gravimetric capacity calculated for', Inputs.grav_cap_method, 'mass')
    print('Gravimetric capacity stored:', np.round(SV_ch.iloc[-1,-1], 2), 'A-h/kg')
    print('Gravimetric capacity recovered:', np.round(SV_dch.iloc[-1,-1], 2), 'A-h/kg', '\n')
    
    print('Areal Capacity stored:', Cap_stored, 'A-h/m^2')
    print('Areal Capacity recovered:', Cap_recovered, 'A-h/m^2')
    print('Coulombic Efficiency:', Eta_c, '\n')
    
# Calculate battery energy storage/recovery and calculate round-trip
#   efficiency. Anode voltage is referenced to its initial equilibrium
#   value (i.e. in the discharged state).

# NOTE: This is in W-h/m^2, per unit area of battery. For the specific
#   capacity, you want W-h/g of anode material.
    E_st = 0
    E_rec = 0

    for k in np.arange(1, len(t_ch)):
        E_st = E_st - 0.5*(V_ch[k] + V_ch[k-1])*i_ext*(dt_ch[k] - dt_ch[k-1])

    for k in np.arange(1, len(t_dch)):
        E_rec = E_rec - 0.5*(V_dch[k]+V_dch[k-1])*i_ext*(dt_dch[k]-dt_dch[k-1])
    
    
    Eta_RT = round(100*E_rec/E_st, 1)
    E_st = round(E_st, 2)
    E_rec = round(E_rec, 2)

    print('Energy stored:', E_st, 'W-h/m^2')
    print('Energy recovered:', E_rec, 'W-h/m^2')
    print('Energetic efficiency:', Eta_RT, '\n')
    
    return

"""========================================================================="""

def tag_strings(SV):
    # Define labels (aka 'tags') for plotting.
    SV_eq_labels = SV.columns.values.tolist()

    X_ca = []
    rho_el_an = []
    rho_el_ca = []
    V_an = []
    V_ca = []
    V_sep = np.array([])
    X_el_sep = []
    
    ptr = an.ptr
    for j in np.arange(0, an.npoints):
        offset = int(an.offsets[j])
        V_an[0+offset:1+offset] = \
            SV_eq_labels[ptr['Phi_ed']+offset:ptr['Phi_dl']+offset+1]
            
        rho_el_an[0+offset:elyte_obj.n_species+offset] = \
            SV_eq_labels[ptr['X_k_elyte'][0]+offset:ptr['X_k_elyte'][-1]+offset+1]
                
    ptr = ca.ptr
    for j in np.arange(0, ca.npoints):
        offset = int(ca.offsets[j])
        V_ca[0+offset:1+offset] = \
            SV_eq_labels[ptr['Phi_ed']+offset:ptr['Phi_dl']+offset+1]
        
        X_ca[0+offset:ca.nshells+offset] = \
            SV_eq_labels[0+offset:ca.nshells+offset]
        
        rho_el_ca[0+offset:elyte_obj.n_species+offset] = \
            SV_eq_labels[ptr['X_k_elyte'][0]+offset:ptr['X_k_elyte'][-1]+offset+1]
            
    ptr = sep.ptr
    for j in np.arange(0, sep.npoints):
        offset = int(sep.offsets[j])
        V_sep = np.append(V_sep, SV_eq_labels[ptr['Phi']+offset])
        X_el_sep[0+offset:elyte_obj.n_species+offset] = \
            SV_eq_labels[ptr['X_k_elyte'][0]+offset:ptr['X_k_elyte'][-1]+offset+1]
    
    V_sep = V_sep.tolist()
    tags = {}
    tags['Phi_an'] = V_an; tags['Phi_ca'] = V_ca; tags['X_ca'] = X_ca
    tags['X_el_an'] = rho_el_an; tags['X_el_ca'] = rho_el_ca; tags['X_el_sep'] = X_el_sep
    tags['Phi_sep'] = V_sep

    return tags

"""========================================================================="""

def Label_Columns(t, SV, anode_np, sep_np, ca_np):
    # Label the pandas dataframe columns.

    # Convert t and SV into pandas data frames
    t_df = pd.DataFrame(t)
    SV_df = pd.DataFrame(SV)
    
    # Set the column label for the t data frame to the number of columns in SV
    newcols_t = {0: SV_df.shape[1]}
    t_df.rename(columns = newcols_t, inplace = True)
    
    # Concatenate t_df onto end of SV_df by columns (axis = 1)
    SV_df = pd.concat((SV_df, t_df), axis = 1)
    
    """Label anode points"""
    newcols = {}
    for j in np.arange(0, anode_np):
        offset = an.offsets[j]  # Set node offset value for loop
            
        # Loop over number of species in electrolyte
        for k in np.arange(0, elyte_obj.n_species):
            species = elyte_obj.species_names[k]
            newcols_el = {k + offset: 'X_'+species+'_an'+str(j+1)}
            newcols.update(newcols_el)
            
        # Add tags for electrode and double layer potentials
        newcols_phi = {0+elyte_obj.n_species+offset: 'Phi_an'+str(j+1),
                       1+elyte_obj.n_species+offset: 'Phi_an_dl'+str(j+1)}
        newcols.update(newcols_phi)
        
        SV_df.rename(columns=newcols, inplace=True)
        
    """Label separator points"""
    newcols = {}
    for j in np.arange(0, sep_np):
        offset = sep.offsets[j] # Set node offset value for loop
        
        # Loop over number of species in electrolyte
        for k in np.arange(0, elyte_obj.n_species):
            species = elyte_obj.species_names[k]
            newcols_el = {k + offset: 'X_'+species+'_sep'+str(j+1)}
            newcols.update(newcols_el)
            
        # Add tag for electrolyte potential
        newcols_phi = {0+elyte_obj.n_species+offset: 'Phi_sep'+str(j+1)}
        newcols.update(newcols_phi)
        
        SV_df.rename(columns=newcols, inplace = True)
    
    """Label cathode points"""
    newcols = {}
    for j in np.arange(0, ca_np):
        offset = ca.offsets[j]  # Set node offset value for loop
        
        # Loop over number of shells in anode
        for k in np.arange(0, ca.nshells):
            newcols_ca = {k + offset: 'X_ca'+str(j+1)+str(k+1)}
            newcols.update(newcols_ca)
            
        # Loop over number of species in electrolyte
        for k in np.arange(0, elyte_obj.n_species):
            species = elyte_obj.species_names[k]
            newcols_el = {k + ca.nshells + offset: 'X_'+species+'_ca'+str(j+1)}
            newcols.update(newcols_el)
            
        # Add tags for electrode and double layer potentials
        newcols_phi = {0+ca.nshells+elyte_obj.n_species+offset:'Phi_ca'+str(j+1),
                       1+ca.nshells+elyte_obj.n_species+offset: 'Phi_ca_dl'+str(j+1)}
        newcols.update(newcols_phi)
        
        SV_df.rename(columns=newcols, inplace=True)
    
    
    newcols_time = {SV_df.shape[1]-1: 'Time'}
    SV_df.rename(columns=newcols_time, inplace=True)

    return SV_df

