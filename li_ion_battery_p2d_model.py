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
from matplotlib import pyplot as plt

from assimulo.solvers import IDA

import li_ion_battery_p2d_functions
importlib.reload(li_ion_battery_p2d_functions)
from li_ion_battery_p2d_functions import Extended_Problem

import li_ion_battery_p2d_inputs
importlib.reload(li_ion_battery_p2d_inputs)
from li_ion_battery_p2d_inputs import Inputs as inp

import li_ion_battery_p2d_init
importlib.reload(li_ion_battery_p2d_init)
from li_ion_battery_p2d_init import anode, cathode, separator, solver_inputs, current

import li_ion_battery_p2d_post_process
importlib.reload(li_ion_battery_p2d_post_process)
from li_ion_battery_p2d_post_process import Label_Columns, tag_strings, plot_sims, plot_cap

def main():
    
    SV_0 = solver_inputs.SV_0
    algvar = solver_inputs.algvar

    # Close any open pyplot objects:
    plt.close('all')
    
    atol1 = 1e-3; atol2 = atol1; atol3 = atol1; atol4 = atol1
    rtol1 = 1e-15; rtol2 = rtol1; rtol3 = rtol1; rtol4 = rtol1    

    # Start a timer:
    t_count = time.time()

    # Calculate the time span, which should be enough to charge or discharge fully
    #   (i.e. 3600 seconds divided by the C-rate):
    t_0 = 0
    t_f = 3600/inp.C_rate
    SV_dot_0 = np.zeros_like(SV_0)
    
    rate_tag = str(inp.C_rate)+"C"
    
    if inp.plot_profiles_flag == 1:
        fig1, axes1 = plt.subplots(sharey = "row", figsize = (18, 9), nrows = inp.n_comps+2, ncols = 2+inp.flag_re_equil)
        plt.subplots_adjust(wspace = 0.15, hspace = 0.4)
        fig1.text(0.15, 0.8, rate_tag, fontsize=20, bbox=dict(facecolor='white', alpha=0.5))
        
    """----------Equilibration----------"""

    # Equilibrate by integrating at zero current:
    print('\nEquilibrating...')

    # Create problem instance
    current.set_i_ext(0)
    
    Battery_equil = Extended_Problem(Extended_Problem.Battery_Func, SV_0, SV_dot_0, t_0)
    Battery_equil.external_event_detection = True
    Battery_equil.algvar = algvar

    # Simulation parameters
    equil_sim = IDA(Battery_equil)           # Create simulation instance
    equil_sim.atol = atol1                 # Solver absolute tolerance
    equil_sim.rtol = rtol1                  # Solver relative tolerance
    equil_sim.verbosity = 50
    equil_sim.make_consistent('IDA_YA_YDP_INIT')

    t_eq, SV_eq, SV_dot_eq = equil_sim.simulate(t_f)

    # Put solution into pandas dataframe with labeled columns
    SV_eq_df = Label_Columns(t_eq, SV_eq, anode.npoints, separator.npoints, 
                             cathode.npoints)

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
    Battery_charge = Extended_Problem(Extended_Problem.Battery_Func, SV_0, SV_dot_0, t_0)
    Battery_charge.external_event_detection = True
    Battery_charge.algvar = algvar

    # Simulation parameters
    charge_sim = IDA(Battery_charge)
    charge_sim.atol = atol2
    charge_sim.rtol = rtol2
    charge_sim.verbosity = 50
    charge_sim.make_consistent('IDA_YA_YDP_INIT')

    t_charge, SV_charge, SV_dot_charge = charge_sim.simulate(t_f)

    if hasattr(anode, 't_flag'):
        t_flag_ch = anode.get_tflag()
    else:
        t_flag_ch = t_charge[-1]

    SV_charge_df = Label_Columns(t_charge, SV_charge, anode.npoints, separator.npoints, 
                             cathode.npoints)
    
    if inp.plot_profiles_flag == 1:
        plot_sims(tags['Phi_an'], tags['Phi_cat'], tags['X_an'], tags['X_cat'],
                  tags['rho_el_an'], tags['rho_el_cat'], SV_charge_df, 
                  'Charging', 1, fig1, axes1)

    print('Done charging\n')

    """------------Re_equilibrating-------------"""
    
    if inp.flag_re_equil == 1:
        # New initial conditions are the final charge conditions
        SV_0 = SV_charge[-1, :]
        SV_dot_0 = SV_dot_charge[-1, :]
    
        # Equilibrate again. Note - this is a specific choice to reflect
        #   equilibration after the charging steps. We may want, at times, to
        #   simulate a situation where the battery is not equilibrated between
        #   charge and discharge, or is equilibrated for a shorter amount of time.
    
        print('\nRe-equilibrating...')
        
        current.set_i_ext(0)
    
        Battery_re_equil = Extended_Problem(Extended_Problem.Battery_Func, SV_0, SV_dot_0, t_0)
        Battery_re_equil.external_event_detection = True
        Battery_re_equil.algvar = algvar
    
        # Simulation parameters
        re_equil_sim = IDA(Battery_re_equil)
        re_equil_sim.atol = atol3
        re_equil_sim.rtol = rtol3
        re_equil_sim.verbosity = 50
        re_equil_sim.make_consistent('IDA_YA_YDP_INIT')
    
        t_req, SV_req, SV_dot_req = re_equil_sim.simulate(t_f)
    
        SV_req_df = Label_Columns(t_req, SV_req, anode.npoints, separator.npoints, 
                             cathode.npoints)
        
        if inp.plot_profiles_flag == 1:
            plot_sims(tags['Phi_an'], tags['Phi_cat'], tags['X_an'], tags['X_cat'], 
                      tags['rho_el_an'], tags['rho_el_cat'], SV_req_df, 
                     'Re-equilibrating', 2, fig1, axes1)
    
        print('Done re-equilibrating\n')
    else:
        SV_req = SV_charge
        SV_dot_req = SV_dot_charge
        
        SV_req_df = SV_req

    """------------Discharging-------------"""

    print('\nDischarging...')

    SV_0 = SV_req[-1, :]
    SV_dot_0 = SV_dot_req[-1, :]

    current.set_i_ext(-current.i_ext_set)

    Battery_discharge = Extended_Problem(Extended_Problem.Battery_Func, SV_0, SV_dot_0, t_0)
    Battery_discharge.external_event_detection = True
    Battery_discharge.algvar = algvar

    # Simulation parameters
    Battery_discharge = IDA(Battery_discharge)
    Battery_discharge.atol = atol4
    Battery_discharge.rtol = rtol4
    Battery_discharge.verbosity = 50
    Battery_discharge.make_consistent('IDA_YA_YDP_INIT')

    t_discharge, SV_discharge, SV_dot_discharge = Battery_discharge.simulate(t_f)

    if hasattr(anode, 't_flag'):
        t_flag_dch = anode.get_tflag()
    else:
        t_flag_dch = t_discharge[-1]

    SV_discharge_df = Label_Columns(t_discharge, SV_discharge, anode.npoints, separator.npoints, 
                             cathode.npoints)
    
    if inp.plot_profiles_flag == 1:
        plot_sims(tags['Phi_an'], tags['Phi_cat'], tags['X_an'], tags['X_cat'],
                  tags['rho_el_an'], tags['rho_el_cat'], SV_discharge_df, 
                 'Discharging', 2+inp.flag_re_equil, fig1, axes1)

    print('Done discharging\n')

    """---------------------------------"""
    
# %% Plot capacity if flagged
    
    if inp.plot_profiles_flag == 1:
        plt.show()
    
    if inp.plot_cap_flag == 1:
        Cap_recovered, Eta_c = plot_cap(SV_charge_df, SV_discharge_df, t_flag_ch,
                                        t_flag_dch, rate_tag, current.i_ext_set)
        print('Cap_recovered = ', Cap_recovered, '\n')
        print('Eta_c = ', Eta_c, '\n')

    elapsed = time.time() - t_count
    print('t_cpu=', elapsed, '\n')
    plt.show()
    
    return SV_eq_df, SV_charge_df, SV_req_df, SV_discharge_df

if __name__ == "__main__":
    SV_eq_df, SV_charge_df, SV_req_df, SV_discharge_df = main()
