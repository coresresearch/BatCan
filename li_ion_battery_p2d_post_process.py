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


def plot_sims(V_an, V_cat, X_an, X_cat, rho_k_an, rho_k_cat, SV_df, stage, yax, fig, axes):
    
    if stage == 'Discharging':
        showlegend = 1
    else:
        showlegend = 0
    showlegend = 0
    fontsize = 12
    
    t = SV_df['Time']
    index = []
    for j in np.arange(0, anode.nshells):
        offset = j*anode.nshells
        index_add = [0+offset, 4+offset]
        index = np.append(index, index_add)
        
    X_an = [X_an[i] for i in index.astype(int)]
    
#    index = []
#    for j in np.arange(0, cathode.nshells):
#        offset = j*cathode.nshells
#        index_add = [0+offset, 4+offset]
#        index = np.append(index, index_add)
#        
#    X_cat = [X_cat[i] for i in index.astype(int)]
    
    index = []
    for i in np.arange(0, anode.npoints):
        offset = i*elyte_obj.n_species
        index_add = [2 + offset]
        index = np.append(index, index_add)
        
    rho_Li_an = [rho_k_an[i] for i in index.astype(int)]
    
    index = []
    for i in np.arange(0, cathode.npoints):
        offset = i*elyte_obj.n_species
        index_add = [2 + offset]
        index = np.append(index, index_add)

    rho_Li_cat = [rho_k_cat[i] for i in index.astype(int)]
    
#    rho_Li = np.concatenate((rho_Li_an, rho_Li_cat))
    
    index = []
    for i in np.arange(0, cathode.npoints):
        index_add = [2*i]
        index = np.append(index, index_add)
        
    V = [V_cat[i] for i in index.astype(int)]
    
    yax = yax - 1
    
#    line_style = ['v-', 'o-', '^-', 's-', 'h-', '+-']
    
    # Plot anode and double-layer potential
    SV_plot = SV_df.plot(x='Time', y=V_cat, ax=axes[0, yax], xlim=[0,t.iloc[-1]])
    SV_plot.set_title(stage, fontsize = fontsize)
    SV_plot.set_ylabel('Voltages [V]', fontsize = fontsize)
    SV_plot.set_xlabel('Time [s]', fontsize = fontsize).set_visible(showlegend)
    SV_plot.legend(loc=2, bbox_to_anchor=(1.05, 1), ncol=1, borderaxespad=0,
                   frameon=False).set_visible(showlegend)
#    SV_plot.tick_params(axis='both', labelsize=18)
    SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    line_style = ['r--', 'r', 'b--', 'b', 'k--', 'k', 'g--', 'g', 'y--', 'y']
    
    # Plot anode composition
    SV_plot = SV_df.plot(x = 'Time', y = X_an, ax = axes[1, yax], xlim = [0, t.iloc[-1]],
                         ylim = [-0.1, 1.1], style = line_style)
#    SV_plot.set_title(stage, fontsize = fontsize)
    SV_plot.set_ylabel('$[X_{LiC_6}]$', fontsize = fontsize)
    SV_plot.set_xlabel('Time [s]', fontsize = fontsize).set_visible(showlegend)
    SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, 
                   borderaxespad = 0, frameon = False).set_visible(showlegend)
#    SV_plot.tick_params(axis='both', labelsize = 18)
    SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Plot cathode composition
    SV_plot = SV_df.plot(x = 'Time', y = X_cat, ax = axes[2, yax], xlim = [0, t.iloc[-1]],
                         ylim = [-0.1, 1.1], style = line_style)
#    SV_plot.set_title(stage, fontsize = fontsize)
    SV_plot.set_ylabel('$[X_{LiCoO2}]$', fontsize = fontsize)
    SV_plot.set_xlabel('Time [s]', fontsize = fontsize).set_visible(0)
    SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, 
                   borderaxespad = 0, frameon = False).set_visible(showlegend)
#    SV_plot.tick_params(axis='both', labelsize = 18)
    SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    if stage == 'Discharging':
        showlegend = 1
    else:
        showlegend = 0
    
    # Plot elyte composition
    SV_plot = SV_df.plot(x = 'Time', y = rho_Li_an, ax = axes[3, yax], xlim = [0, t.iloc[-1]])
    SV_plot.set_ylabel(r'$\left[\frac{kmol_k}{m^3}\right]_{anode}$', fontsize = fontsize)
    SV_plot.set_xlabel('Time [s]', fontsize = fontsize)
    SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, borderaxespad = 0,
                   frameon = False).set_visible(showlegend)
    SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    SV_plot = SV_df.plot(x = 'Time', y = rho_Li_cat, ax = axes[4, yax], xlim = [0, t.iloc[-1]])
    SV_plot.set_ylabel(r'$\left[\frac{kmol_k}{m^3}\right]_{cathode}$', fontsize = fontsize)
    SV_plot.set_xlabel('Time [s]', fontsize = fontsize)
    SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, borderaxespad = 0,
                   frameon = False).set_visible(showlegend)
    SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

"""========================================================================="""

def plot_cap(SV_ch_df, SV_dch_df, t_flag_ch, t_flag_dch, rate_tag, i_ext):
    fontsize = 18
    
    SV_ch_df = SV_ch_df.loc[SV_ch_df['Time'] <= t_flag_ch-1]
    SV_dch_df = SV_dch_df.loc[SV_dch_df['Time'] <= t_flag_dch-1]
    
    V_charge = np.array(SV_ch_df['Phi_an_dl1'])
    V_discharge = np.array(SV_dch_df['Phi_an_dl1'])
    t_charge = np.array(SV_ch_df['Time'])
    t_discharge = np.array(SV_dch_df['Time'])
    dt_charge = t_charge - t_charge[0]
    dt_discharge = t_discharge - t_discharge[0]
    
    # Plot charge-discharge curve
    Capacity_charge = -dt_charge*i_ext/3600         # A-h/m^2
    Capacity_discharge = -dt_discharge*i_ext/3600   # A-h/m^2
    
    plt.figure(2, figsize = (8, 6))
    plt.plot(Capacity_charge, V_charge, 'b-')
    plt.plot(Capacity_discharge, V_discharge, 'b--')
    plt.xlim((-0.5, 30))
#    plt.ylim((1, 4))
    plt.title('Split-cell potential vs. Capacity', fontsize = fontsize)
    plt.xlabel('$Capacity [Ah/m^2]$', fontsize = fontsize)
    plt.ylabel('Voltage [V]', fontsize = fontsize)
    plt.legend(('Charge', 'Discharge'), loc = 3, fontsize = 14)
    plt.show()
    
#    plt.figure(1, figsize = (8, 6))
#    ax1 = plt.subplot(1, 1, 1)
#    plt.plot(Capacity_charge, V_charge, 'r-')
#    plt.plot(Capacity_discharge, V_discharge, 'r--')
#    ax1.set_ylim((0, 0.3))
#    ax1.set_xlim((-0.1, 20))
#    plt.title('Half-cell potential vs. Capacity', fontsize = fontsize)
#    plt.xlabel('$Capacity [Ah/m^2]$', fontsize = fontsize)
#    plt.ylabel('Voltage [V]', fontsize = fontsize)
#    plt.legend(('Charge capacity', 'Discharge capacity'), loc=3, fontsize = 16)
    
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
    
    ptr = anode.ptr
    for j in np.arange(0, anode.npoints):
        offset = int(anode.offsets[j])
        V_an[0+offset:1+offset] = \
            SV_eq_labels[ptr['Phi_ed']+offset:ptr['Phi_dl']+offset+1]
            
        X_an[0+offset:anode.nshells+offset] = \
            SV_eq_labels[0+offset:anode.nshells+offset]
            
        rho_el_an[0+offset:elyte_obj.n_species+offset] = \
            SV_eq_labels[ptr['rho_k_elyte'][0]+offset:ptr['rho_k_elyte'][-1]+offset+1]
                
    ptr = cathode.ptr
    for j in np.arange(0, cathode.npoints):
        offset = int(cathode.offsets[j])
        V_cat[0+offset:1+offset] = \
            SV_eq_labels[ptr['Phi_ed']+offset:ptr['Phi_dl']+offset+1]
        
        X_cat[0+offset:cathode.nshells+offset] = \
            SV_eq_labels[0+offset:cathode.nshells+offset]
        
        rho_el_cat[0+offset:elyte_obj.n_species+offset] = \
            SV_eq_labels[ptr['rho_k_elyte'][0]+offset:ptr['rho_k_elyte'][-1]+offset+1]

    tags = {}
    tags['Phi_an'] = V_an; tags['Phi_cat'] = V_cat; tags['X_an'] = X_an; tags['X_cat'] = X_cat
    tags['rho_el_an'] = rho_el_an; tags['rho_el_cat'] = rho_el_cat

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
            newcols_el = {k + anode.nshells + offset: 'rho_'+species+'_an'+str(j+1)}
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
            newcols_el = {k + offset: 'rho_'+species+'_sep'+str(j+1)}
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
            newcols_el = {k + cathode.nshells + offset: 'rho_'+species+'_cat'+str(j+1)}
            newcols.update(newcols_el)
            
        # Add tags for electrode and double layer potentials
        newcols_phi = {0+cathode.nshells+elyte_obj.n_species+offset: 'Phi_cat'+str(j+1),
                       1+cathode.nshells+elyte_obj.n_species+offset: 'Phi_cat_dl'+str(j+1)}
        newcols.update(newcols_phi)
        
        SV_df.rename(columns=newcols, inplace=True)
    
    
    newcols_time = {SV_df.shape[1]-1: 'Time'}
    SV_df.rename(columns=newcols_time, inplace=True)

    return SV_df
