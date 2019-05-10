# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:59:27 2018

@author: dkorff
"""

import importlib
import li_ion_battery_p2d_init
importlib.reload(li_ion_battery_p2d_init)
from li_ion_battery_p2d_init import anode, cathode
from li_ion_battery_p2d_init import separator as sep
from li_ion_battery_p2d_init import anode_obj, cathode_obj, elyte_obj
from li_ion_battery_p2d_init import current

import li_ion_battery_p2d_inputs
importlib.reload(li_ion_battery_p2d_inputs)
from li_ion_battery_p2d_inputs import Inputs

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from math import floor, ceil

def plot_potential(V_an, V_cat, SV_df, stage, yax, t_flag, fig, axes):
    
    if stage == 'Discharging':
        showlegend = 1
    else:
        showlegend = 0
    fontsize = 12
    
#    SV_df = SV_df.loc[SV_df['Time'] <= t_flag-1]
    t = SV_df['Time']
    
    if Inputs.flag_cathode:
        index = []
        for i in np.arange(0, cathode.npoints):
            index_add = [2*i]
            index = np.append(index, index_add)
            
        V = [V_cat[i] for i in index.astype(int)]
#        SOC = SV_df.filter(like="X_cat").sum(1)/(cathode.npoints*cathode.nshells)
        
    elif not Inputs.flag_cathode:
        index = []
        for i in np.arange(0, anode.npoints):
            index_add = [1 + 2*i]
            index = np.append(index, index_add)
            
        V = [V_an[i] for i in index.astype(int)]
            
    #    line_style = ['v-', 'o-', '^-', 's-', 'h-', '+-']
        
    # Plot cathode and double-layer potential
    SV_plot = SV_df.plot(x='Time', y=V, ax=axes[yax], xlim=[0,t.iloc[-1]])
    SV_plot.set_title(stage, fontsize = fontsize)
    SV_plot.set_ylabel('Voltages [V]', fontsize = fontsize)
    SV_plot.set_xlabel('Time [s]', fontsize = fontsize).set_visible(showlegend)
    SV_plot.legend(loc=2, bbox_to_anchor=(1.05, 1), ncol=1, borderaxespad=0,
                   frameon=False).set_visible(showlegend)
    SV_plot.tick_params(axis='both', labelsize=18)
    SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

"""========================================================================="""

def plot_electrode(X_an, X_cat, SV_df, stage, yax, fig, axes):
    
    t = SV_df['Time']
    fontsize = 26
    
    if stage == 'Discharging':
        showlegend = 1
    else:
        showlegend = 0
    
    if Inputs.flag_anode and Inputs.flag_cathode:
        axes_an = axes[0, yax]
        axes_cat = axes[1, yax]
    elif Inputs.flag_anode and not Inputs.flag_cathode:
        axes_an = axes[yax]
    elif Inputs.flag_cathode and not Inputs.flag_anode:
        axes_cat = axes[yax]
    
    if Inputs.flag_anode:
        index = []
        for i in np.arange(0, anode.npoints):
            offset = i*anode.nshells
            index_add = [0+offset, anode.nshells-1+offset]
            index = np.append(index, index_add)
            
        X_an = [X_an[i] for i in index.astype(int)]
        
        # Plot anode composition
        SV_plot = SV_df.plot(x = 'Time', y = X_an, ax = axes_an,
                             xlim = [0, t.iloc[-1]], ylim = [-0.1, 1.1])
#        SV_plot.set_title(stage, fontsize = fontsize)
        SV_plot.set_ylabel('$X_{LiC_6}$', fontsize = fontsize)
        SV_plot.set_xlabel('Time [s]', fontsize = fontsize).set_visible(True)
        SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, 
                       borderaxespad = 0, frameon = False).set_visible(False)
#        SV_plot.legend((str(rate_tag)), loc = 3, fontsize = 14)
        SV_plot.tick_params(axis='both', labelsize = 18)
        SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    if Inputs.flag_cathode:
        index = []
        for i in np.arange(0, cathode.npoints):
            offset = i*cathode.nshells
            index_add = [0+offset, cathode.nshells-1+offset]
            index = np.append(index, index_add)
        
        X_cat_1 = [X_cat[i] for i in index.astype(int)]
    
        # Plot cathode composition
        SV_plot = SV_df.plot(x = 'Time', y = X_cat_1, ax = axes_cat, xlim = [0, t.iloc[-1]],
                             ylim = [-0.1, 1.1])
        SV_plot.set_title(stage, fontsize = fontsize)
        SV_plot.set_ylabel('$[X_{LiCoO2}]$', fontsize = fontsize)
        SV_plot.set_xlabel('Time [s]', fontsize = fontsize).set_visible(0)
        SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, 
                       borderaxespad = 0, frameon = False).set_visible(showlegend)
        SV_plot.tick_params(axis='both', labelsize = 18)
        SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    return

"""========================================================================="""

def plot_elyte(rho_k_an, rho_k_cat, rho_k_sep, SV_df, stage, yax, fig, axes):
    
    t = SV_df['Time']
    fontsize = 12
    
    if stage == 'Discharging':
        showlegend = 1
    else:
        showlegend = 0
        
    axes_an = axes[0, yax]
    axes_sep = axes[Inputs.flag_sep, yax]
    axes_cat = axes[Inputs.flag_sep+Inputs.flag_cathode, yax]
    
    if Inputs.flag_anode:
        index = []
        for i in np.arange(0, anode.npoints):
            offset = i*elyte_obj.n_species
            index_add = [2+offset]
            index = np.append(index, index_add)
            
        rho_Li_an = [rho_k_an[i] for i in index.astype(int)]
    
    if Inputs.flag_cathode:
        index = []
        for i in np.arange(0, cathode.npoints):
            offset = i*elyte_obj.n_species
            index_add = [2+offset]
            index = np.append(index, index_add)
            
        rho_Li_cat = [rho_k_cat[i] for i in index.astype(int)]
        
    if Inputs.flag_sep:
        index = []
        for i in np.arange(0, sep.npoints):
            offset = i*elyte_obj.n_species
            index_add = [2+offset]
            index = np.append(index, index_add)
            
        rho_Li_sep = [rho_k_sep[i] for i in index.astype(int)]
    
    if Inputs.flag_anode and Inputs.flag_cathode:
        max_val = max(max(SV_df[rho_Li_an].max()), max(SV_df[rho_Li_cat].max()))
#        ymax = ceil(max_val) - (ceil(max_val) - max_val)*0.9
        ymax = max_val + max_val/10
        min_val = min(min(SV_df[rho_Li_an].min()), min(SV_df[rho_Li_cat].min()))
#        ymin = floor(min_val) + (min_val - floor(min_val))*0.9
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
        SV_plot.set_ylabel(r'$X_{Li^+, an}$', fontsize = fontsize)  # \left[\frac{kmol_k}{m^3}\right]_{anode}
        SV_plot.set_xlabel('Time [s]', fontsize = fontsize)
        SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, borderaxespad = 0,
                       frameon = False).set_visible(showlegend)
        SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    if Inputs.flag_cathode:
        # Plot elyte composition in cathode
        SV_plot = SV_df.plot(x = 'Time', y = rho_Li_cat, ax = axes_cat, 
                             xlim = [0, t.iloc[-1]], ylim = [ymin, ymax])
        SV_plot.set_title(stage, fontsize = fontsize)
        SV_plot.set_ylabel(r'$X_{Li^+, cat}$', fontsize = fontsize)  # \left[\frac{kmol_k}{m^3}\right]_{cathode}
        SV_plot.set_xlabel('Time [s]', fontsize = fontsize)
        SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, borderaxespad = 0,
                       frameon = False).set_visible(showlegend)
        SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
    if Inputs.flag_sep:
        # Plot elyte composition in separator
        SV_plot = SV_df.plot(x = 'Time', y = rho_Li_sep, ax = axes_sep, 
                             xlim = [0, t.iloc[-1]], ylim = [ymin, ymax])
        SV_plot.set_title(stage, fontsize = fontsize)
        SV_plot.set_ylabel(r'$X_{Li^+, sep}$', fontsize = fontsize)  # \left[\frac{kmol_k}{m^3}\right]_{cathode}
        SV_plot.set_xlabel('Time [s]', fontsize = fontsize)
        SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, borderaxespad = 0,
                       frameon = False).set_visible(showlegend)
        SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    return

"""========================================================================="""

def plot_cap(SV_ch_df, SV_dch_df, t_flag_ch, t_flag_dch, rate_tag, i_ext):
    fontsize = 18
    
    SV_ch_df = SV_ch_df.loc[SV_ch_df['Time'] <= t_flag_ch-1]
    SV_dch_df = SV_dch_df.loc[SV_dch_df['Time'] <= t_flag_dch-1]
    
    if Inputs.flag_cathode:
        V_charge = np.array(SV_ch_df['Phi_cat5'])
        V_discharge = np.array(SV_dch_df['Phi_cat5'])
        t_charge = np.array(SV_ch_df['Time'])
        t_discharge = np.array(SV_dch_df['Time'])
        dt_charge = t_charge - t_charge[0]
        dt_discharge = t_discharge - t_discharge[0]
    elif Inputs.flag_anode and not Inputs.flag_cathode:
        V_charge = np.array(SV_ch_df['Phi_an_dl3'])
        V_discharge = np.array(SV_dch_df['Phi_an_dl3'])
        t_charge = np.array(SV_ch_df['Time'])
        t_discharge = np.array(SV_dch_df['Time'])
        dt_charge = t_charge - t_charge[0]
        dt_discharge = t_discharge - t_discharge[0]
    
    # Plot charge-discharge curve
    Capacity_charge = -dt_charge*i_ext/3600         # A-h/m^2
    Capacity_discharge = -dt_discharge*i_ext/3600   # A-h/m^2
    
    plt.figure(2, figsize = (8, 6))
    plt.plot(Capacity_charge, V_charge, 'g-')
    plt.plot(Capacity_discharge, V_discharge, 'g--')
    plt.title('Potential vs. Capacity', fontsize = fontsize)
    plt.xlabel('$Capacity [Ah/m^2]$', fontsize = fontsize)
    plt.ylabel('Voltage [V]', fontsize = fontsize)
    plt.legend(('Charge', 'Discharge'), loc = 3, fontsize = 14)
    plt.show()
    
    Cap_recovered = round(Capacity_discharge[-1], 2)
    Cap_stored = Capacity_charge[-1]
    Eta_c = round(100*Cap_recovered/Cap_stored, 1)
    
#    ax1.text(0.01, 0.01, r"$\eta_c$="+str(Eta_c)+"% at "+str(rate_tag), 
#             fontsize = fontsize)
    
# Calculate battery energy storage/recovery and calculate round-trip
#   efficiency. Anode voltage is referenced to its initial equilibrium
#   value (i.e. in the discharged state).

# NOTE: This is in W-h/m^2, per unit area of battery. For the specific
#   capacity, you want W-h/g of anode material.
#    E_stored = 0
#    E_recovered = 0
#
#    for k in np.arange(1, len(t_charge)):
#        E_stored = (E_stored - (anode.V_cathode - 0.5*(V_charge[k] + V_charge[k-1]))
#                    *ep.i_ext*(dt_charge[k] - dt_charge[k-1]))
#
#    for k in np.arange(1, len(t_discharge)):
#        E_recovered = (E_recovered - (ep.V_cathode -
#                        0.5*(V_discharge[k] + V_discharge[k-1]))
#                        *ep.i_ext*(dt_discharge[k] - dt_discharge[k-1]))
#
#    Cap_recovered = Capacity_discharge[-1]
#    Eta_RT = E_recovered/E_stored
#
#    
#    print(E_stored, '\n')
#    print(E_recovered, '\n')
    
    return Cap_recovered, Eta_c

"""========================================================================="""

def tag_strings(SV):
    SV_eq_labels = SV.columns.values.tolist()

    X_an = []
    X_cat = []
    rho_el_an = []
    rho_el_cat = []
    V_an = []
    V_cat = []
    V_sep = np.array([])
    X_el_sep = []
    
    ptr = anode.ptr
    for j in np.arange(0, anode.npoints):
        offset = int(anode.offsets[j])
        V_an[0+offset:1+offset] = \
            SV_eq_labels[ptr['Phi_ed']+offset:ptr['Phi_dl']+offset+1]
            
        X_an[0+offset:anode.nshells+offset] = \
            SV_eq_labels[0+offset:anode.nshells+offset]
            
        rho_el_an[0+offset:elyte_obj.n_species+offset] = \
            SV_eq_labels[ptr['X_k_elyte'][0]+offset:ptr['X_k_elyte'][-1]+offset+1]
                
    ptr = cathode.ptr
    for j in np.arange(0, cathode.npoints):
        offset = int(cathode.offsets[j])
        V_cat[0+offset:1+offset] = \
            SV_eq_labels[ptr['Phi_ed']+offset:ptr['Phi_dl']+offset+1]
        
        X_cat[0+offset:cathode.nshells+offset] = \
            SV_eq_labels[0+offset:cathode.nshells+offset]
        
        rho_el_cat[0+offset:elyte_obj.n_species+offset] = \
            SV_eq_labels[ptr['X_k_elyte'][0]+offset:ptr['X_k_elyte'][-1]+offset+1]
            
    ptr = sep.ptr
    for j in np.arange(0, sep.npoints):
        offset = int(sep.offsets[j])
        V_sep = np.append(V_sep, SV_eq_labels[ptr['Phi']+offset])
        X_el_sep[0+offset:elyte_obj.n_species+offset] = \
            SV_eq_labels[ptr['X_k_elyte'][0]+offset:ptr['X_k_elyte'][-1]+offset+1]
    
    V_sep = V_sep.tolist()
    tags = {}
    tags['Phi_an'] = V_an; tags['Phi_cat'] = V_cat; tags['X_an'] = X_an; tags['X_cat'] = X_cat
    tags['X_el_an'] = rho_el_an; tags['X_el_cat'] = rho_el_cat; tags['X_el_sep'] = X_el_sep
    tags['Phi_sep'] = V_sep

    return tags

"""========================================================================="""

def Label_Columns(t, SV, anode_np, sep_np, cat_np):
    
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
        offset = anode.offsets[j]  # Set node offset value for loop
        
        # Loop over number of shells in anode
        for k in np.arange(0, anode.nshells):
            newcols_an = {k + offset: 'X_an'+str(j+1)+str(k+1)}
            newcols.update(newcols_an)
            
        # Loop over number of species in electrolyte
        for k in np.arange(0, elyte_obj.n_species):
            species = elyte_obj.species_names[k]
            newcols_el = {k + anode.nshells + offset: 'X_'+species+'_an'+str(j+1)}
            newcols.update(newcols_el)
            
        # Add tags for electrode and double layer potentials
        newcols_phi = {0+anode.nshells+elyte_obj.n_species+offset: 'Phi_an'+str(j+1),
                       1+anode.nshells+elyte_obj.n_species+offset: 'Phi_an_dl'+str(j+1)}
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
    for j in np.arange(0, cat_np):
        offset = cathode.offsets[j]  # Set node offset value for loop
        
        # Loop over number of shells in anode
        for k in np.arange(0, anode.nshells):
            newcols_cat = {k + offset: 'X_cat'+str(j+1)+str(k+1)}
            newcols.update(newcols_cat)
            
        # Loop over number of species in electrolyte
        for k in np.arange(0, elyte_obj.n_species):
            species = elyte_obj.species_names[k]
            newcols_el = {k + cathode.nshells + offset: 'X_'+species+'_cat'+str(j+1)}
            newcols.update(newcols_el)
            
        # Add tags for electrode and double layer potentials
        newcols_phi = {0+cathode.nshells+elyte_obj.n_species+offset: 'Phi_cat'+str(j+1),
                       1+cathode.nshells+elyte_obj.n_species+offset: 'Phi_cat_dl'+str(j+1)}
        newcols.update(newcols_phi)
        
        SV_df.rename(columns=newcols, inplace=True)
    
    
    newcols_time = {SV_df.shape[1]-1: 'Time'}
    SV_df.rename(columns=newcols_time, inplace=True)

    return SV_df
