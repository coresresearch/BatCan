# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:59:27 2018

@author: dkorff
"""

from li_ion_battery_p2d_init import anode, cathode
from li_ion_battery_p2d_init import separator as sep
from li_ion_battery_p2d_init import anode_obj, cathode_obj, elyte_obj
from li_ion_battery_p2d_init import current
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def plot_sims(V, X, rho_k_el, SV_df, stage, yax, fig, axes):
    
    if stage == 'Discharging':
        showlegend = 1
    else:
        showlegend = 0
    fontsize = 12
#    showlegend = 0
    
    t = SV_df['Time']
    index = []
    for j in np.arange(0, anode.nshells):
        offset = j*anode.nshells
        index_add = [0+offset, 4+offset]
        index = np.append(index, index_add)
        
    X = [X[i] for i in index.astype(int)]
    
#    index = []
#    for j in np.arange(0, anode.npoints):
#        offset = j*2
#        index_add = [0+offset, 1+offset]
#        index = np.append(index, index_add)
#        
#    V = [V[i] for i in index.astype(int)]
        
#    index = [0]
#    index_add = np.arange(1, 9+2, 2)
#    index = np.append(index, index_add)
#    V = [V[i] for i in index.astype(int)]
    
    index = []
    for i in np.arange(0, anode.npoints):
        offset = i*elyte_obj.n_species
        index_add = [2 + offset]
        index = np.append(index, index_add)
        
    rho_Li = [rho_k_el[i] for i in index.astype(int)]
    
    yax = yax - 1
    SV_df['Time'] = SV_df['Time']
    
#    line_style = ['v-', 'o-', '^-', 's-', 'h-', '+-']
    
    # Plot anode and double-layer potential
    SV_plot = SV_df.plot(x='Time', y=V, ax=axes[0, yax], xlim=[0,t.iloc[-1]])
    SV_plot.set_title(stage, fontsize = fontsize)
    SV_plot.set_ylabel('Voltages [V]', fontsize = fontsize)
#    SV_plot.set_xlabel('Time [s]', fontsize = fontsize)
    SV_plot.legend(loc=2, bbox_to_anchor=(1.05, 1), ncol=1, borderaxespad=0,
                   frameon=False).set_visible(showlegend)
    SV_plot.tick_params(axis='both', labelsize=18)
    SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    line_style = ['r--', 'r', 'b--', 'b', 'k--', 'k', 'g--', 'g', 'y--', 'y']
    
    # Plot anode composition
    SV_plot = SV_df.plot(x = 'Time', y = X, ax = axes[1, yax], xlim = [0, t.iloc[-1]],
                         ylim = [-0.1, 1.1], style = line_style)
    SV_plot.set_title(stage, fontsize = fontsize)
    SV_plot.set_ylabel('Anode composition $[X_{LiC_6}]$', fontsize = fontsize)
#    SV_plot.set_xlabel('Time [s]', fontsize = fontsize)
    SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, 
                   borderaxespad = 0, frameon = False).set_visible(showlegend)
    SV_plot.tick_params(axis='both', labelsize = 18)
    SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Plot elyte composition
    SV_plot = SV_df.plot(x = 'Time', y = rho_Li, ax = axes[2, yax], xlim = [0, t.iloc[-1]])
    SV_plot.set_ylabel(r'Electrolyte composition [kmol_k/m^3]')
    SV_plot.set_xlabel('Time [s]')
    SV_plot.legend(loc = 2, bbox_to_anchor = (1, 1), ncol = 1, borderaxespad = 0,
                   frameon = False).set_visible(showlegend)
    SV_plot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

"""========================================================================="""

def plot_cap(SV_ch_df, SV_dch_df, t_flag_ch, t_flag_dch, rate_tag):
    fontsize = 18
    
    SV_ch_df = SV_ch_df.loc[SV_ch_df['Time'] <= t_flag_ch]
    SV_dch_df = SV_dch_df.loc[SV_dch_df['Time'] <= t_flag_dch]
    
    V_charge = np.array(SV_ch_df['Phi_dl1'])
    V_discharge = np.array(SV_dch_df['Phi_dl1'])
    t_charge = np.array(SV_ch_df['Time'])
    t_discharge = np.array(SV_dch_df['Time'])
    dt_charge = t_charge - t_charge[0]
    dt_discharge = t_discharge - t_discharge[0]
    
    # Plot charge-discharge curve
    Capacity_charge = -dt_charge*current.i_ext_set/3600         # A-h/m^2
    Capacity_discharge = -dt_discharge*current.i_ext_set/3600   # A-h/m^2
    
    fig1, ax1 = plt.subplots(figsize = (8, 6))
    ax1.plot(Capacity_charge, V_charge)
    ax1.plot(Capacity_discharge, V_discharge)
#    ax1.set_ylim((0, 0.3))
#    ax1.set_xlim((-0.1, 20))
    ax1.set_title('Half-cell potential vs. Capacity', fontsize = fontsize)
    ax1.set_xlabel('$Capacity [Ah/m^2]$', fontsize = fontsize)
    ax1.set_ylabel('Voltage [V]', fontsize = fontsize)
    ax1.legend(('Charge capacity', 'Discharge capacity'), loc=3, fontsize = 16)
    
    Cap_recovered = round(Capacity_discharge[-1], 2)
    Cap_stored = Capacity_charge[-1]
    Eta_c = round(100*Cap_recovered/Cap_stored, 1)
    
    ax1.text(0.01, 0.01, r"$\eta_c$="+str(Eta_c)+"% at "+str(rate_tag), 
             fontsize = fontsize)
    
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

    for j in np.arange(0, anode.npoints):
        offset = int(anode.offsets[j])
        V_an[0+offset:1+offset] = \
            SV_eq_labels[anode.ptr['Phi_ed']+offset:anode.ptr['Phi_dl']+offset+1]
            
        X_an[0+offset:anode.nshells+offset] = \
            SV_eq_labels[0+offset:anode.nshells+offset]
            
        rho_el_an[0+offset:elyte_obj.n_species+offset] = \
            SV_eq_labels[anode.ptr['rho_k_elyte'][0]+offset:anode.ptr['rho_k_elyte'][-1]+offset+1]
            
#        rho_el_an.append(SV_eq_labels[offset + anode.nshells])

    for j in np.arange(0, cathode.npoints):
        offset = int(cathode.offset_vec[j])
        V_cat[0+offset:1+offset] = SV_eq_labels[cathode.ptr['Phi_ed']+offset:cathode.ptr['Phi_dl']+offset+1]
        X_cat[0+offset:cathode.nshells+offset] = SV_eq_labels[0+offset:cathode.nshells+offset]
        rho_el_cat.append(SV_eq_labels[offset + cathode.nshells])

    tags = {}
    tags['Phi_an'] = V_an; tags['Phi_cat'] = V_cat; tags['X_an'] = X_an; tags['X_cat'] = X_cat
    tags['rho_el_an'] = rho_el_an; tags['rho_el_cat'] = rho_el_cat

    return tags

"""========================================================================="""

def Label_Columns(t, SV):
    
    # Convert t and SV into pandas data frames
    t_df = pd.DataFrame(t)
    SV_df = pd.DataFrame(SV)
    
    # Set the column label for the t data frame to the number of columns in SV
    newcols_t = {0: SV_df.shape[1]}
    t_df.rename(columns = newcols_t, inplace = True)
    
    # Concatenate t_df onto end of SV_df by columns (axis = 1)
    SV_df = pd.concat((SV_df, t_df), axis = 1)
    
    newcols = {}
    for j in np.arange(0, anode.npoints):
        offset = anode.offsets[j]  # Set node offset value for loop
        
        # Loop over number of shells in anode
        for k in np.arange(0, anode.nshells):
            newcols_an = {k + offset: 'X_an'+str(j+1)+str(k+1)}
            newcols.update(newcols_an)
            
        # Loop over number of species in electrolyte
        for k in np.arange(0, elyte_obj.n_species):
            species = elyte_obj.species_names[k]
            newcols_el = {k + anode.nshells + offset: 'rho_'+species+str(j+1)}
            newcols.update(newcols_el)
            
        # Add tags for electrode and double layer potentials
        newcols_phi = {0+anode.nshells+elyte_obj.n_species+offset: 'Phi_an'+str(j+1),
                       1+anode.nshells+elyte_obj.n_species+offset: 'Phi_dl'+str(j+1)}
        newcols.update(newcols_phi)
        
        SV_df.rename(columns=newcols, inplace=True)
    
    newcols_time = {SV_df.shape[1]-1: 'Time'}
    SV_df.rename(columns=newcols_time, inplace=True)

    return SV_df
