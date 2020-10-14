# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:27:54 2018

@author: dkorff
"""

import numpy as np
import os

class Inputs():

    # Name a folder (or nested folder) where you want to save the output:
    save_folder = 'Debugging/debug'

    """ The galvanostatic boundary condition is provided either as
    - The C-rate (C_rate), or 
    - The external current (i_ext)"""
    # The C-rate is the rate of charge/discharge - how many charges/discharges
    #   can be carried out in 1 hour? This sets the current density:
    # C_rate = 17.5
    # External current, A/m2 (divide by ten to get mA/cm2)
    i_ext = 10 

    # Separator thickness [m]
    H_elyte = 20e-6

    # Cathode thickness [m]
    H_ca = 100e-6

    # Average particle diameter for LiFePO4 [m]*****
    d_part_ca = 0.4e-6

    # Cathode combined solids volume fraction [-]
    eps_solid_ca = 0.7
   
    # String to set the kinetics method used for the electrodes interface. 
    #    Options are: MH, BV, and MHC. MH is Marcus-Hush, BV is Butler-Volmer 
    #    and MHC will use Marcus-Hush-Chidsey theory
    kinetics = 'MHC'

    # These flags specify whether to include each element (anode, separator,
    #   cathode) in the simulation:
    flag_anode = 1
    flag_sep = 1
    flag_cathode = 1
    
    n_comps = flag_anode + flag_sep + flag_cathode
    
    # Number of discretized volumes in the y-direction:
    npoints_anode = 1*flag_anode
    npoints_cathode = 15*flag_cathode
    npoints_elyte = 4*flag_sep
    
    # Flag to allow re-equilibration between charge/discharge
    flag_re_equil = 1
    equil_time = 1e2
    
    """These flags specify whether to plot various data types"""
    # You can turn all profiles on/off with this switch.
    plot_profiles_flag = 1 
    # Plots potential profiles
    plot_potential_profiles = 1*plot_profiles_flag
    # Plots solid phase mole fraction profiles
    plot_electrode_profiles = 1*plot_profiles_flag
    # Plots concentration of Li+ in elyte phase
    plot_elyte_profiles = 1*plot_profiles_flag
    # Plots dis/charge capacity curves
    plot_cap_flag = 1
    
    """Options for plots"""
    # phi_time sets the options for the potential plot. 0 will plot a charge/
    #   discharge profile on a single axes. 1 will plot potential over time
    #   including the re-equilibration step (assuming re-equilibration is
    #   turned on).
    phi_time = 0*plot_potential_profiles
    
    # Set number of charge/discharge cycles to run
    n_cycles = 1
    
    # Flag to determine whether to use a constant dr or constant dv 
    #   discretization of the electrode particles
    particle_method = 'dr'
    
    # Method of calculating gravimetric capacity. Either 'cathode' to normalize
    #   by just the solid cathode material mass or 'cell' to normalize by
    #   the total cell mass
    grav_cap_method = 'cathode'
    
    # Method of calculating capacity. Either 'areal' (geometric area basis) or 
    #   'grav' (gravimetric basis)
    cap_method = 'areal'
        
    # Flag to turn on or off having a constant SEI resistance
    anode_SEI_flag = True
    SEI_tol = 5 # Number of decimals to verify the currents are equal
    R_SEI = 0.0032 # SEI ionic resistance in [Ohm-m2]
    
    # Set the average roughness (actual surface area, divided by geometric area)
    #   for the dense lithium anode. This will affect the effective surface 
    #   area of the lithium anode.
    anode_roughness = 1.
    
    # Set electrolyte transport model to eithe dilute solution ('dst') or
    #   concentrated solution theory ('cst')
    elyte_flux_model = 'dst'
    
    # Simulation temperature (or initial temperature)
    T = 300  # [K]

    # Set initial SOC to generalize both electrode initial lithiation
    # Fully charged = anode fully lithiated, cathode fully de-lithiated.
    # Range of value is from 0 to 1.
    SOC_0 = 0.001

    # Number of "shells" in cathode particle:
    n_shells_cathode = 5

    "Cantera and YAML file info:"
    canterafile = 'lithium_ion_battery_ideal.yml'
    anode_phase = 'anode'
    cathode_phase = 'cathode'
    metal_phase = 'electron'
    elyte_phase = 'electrolyte'
    anode_surf_phase = 'edge_anode_electrolyte'
    cathode_surf_phase = 'edge_cathode_electrolyte'

    Li_species_anode = 'Li(s)'
    Li_species_cathode = 'Li(cathode)'
    Vac_species_cathode = 'V(cathode)'
    
    Li_species_elyte = 'Li+(e)'

    Phi_anode_init = 0.0
    Phi_elyte_init = 2.5
    Delta_Phi_init = 3.4

    # Cutoff Values for lithiation and delithiation of anode:
    Li_an_min = 0.005; Li_an_max = 1 - Li_an_min
    Li_ca_min = 0.005; Li_ca_max = 1 - Li_ca_min

    "Anode geometry and transport"
    # Microstructure
    H_Li = 20e-6        # Lithium foil thickness
    H_an = 22e-6        # Anode thickness [m] - includes Li foil plus a small 
                        #   volume of elyte

    # Other Parameters
    C_dl_an = 1.5e-2    # Double-layer capacitance [F/m^2]
    sigma_an = 75.0     # Bulk anode electrical conductivity [S/m]

    D_Li_an = 7.5e-16   # Bulk diffusion coefficient for Li in graphite [m^2/s]
    D_Li_an_el = np.array([1e-12, 1e-12, 1e-10, 3e-11])

    "Electrolyte geometry and transport"
    rho_sep = 970 # Density of separator material assuming HDPE [kg/m^3]
    # Elyte species bulk diffusion coefficients [m^2/s]
    #TODO #13
    D_Li_elyte = np.array([1e-12, 1e-12, 1.517e-10, 1.517e-10])# 3e-12])
    #TODO #14
    z_k_elyte = np.array([0., 0., 1., -1.])

    eps_elyte_sep = 0.55 # Separator electrolyte volume fraction
    tau_sep = 1.1       # Tortuosity of separator
    sigma_sep = 50.0    # Bulk ionic conductivity of separator [S/m]

    "Cathode geometry and transport"
    # Microstructure:
    wt_pct_active = 0.85 # Weight percent of active material [-]
    wt_pct_cond = 0.10   # Weight percent of conductive additive [-]
    wt_pct_bind = 0.05   # Weight percent of binder (assumed dead material) [-]
    conductor_rho = 2150 # Density of conductor, assumed carbon [kg/m^3]
    binder_rho = 1780    # Density of binder, assumed PVDF [kg/m^3]
    tau_ca = 1.2         # Tortuosity - assume equal values for LFP & elyte [-]
    r_p_ca = 5e-6        # Average pore radius [m]
    overlap_ca = 0.4     # Percentage of cathode particle overlapping with other
                         #   cathode particles.  Reduces total cathode/elyte
                         #   surface area.

    # Other parameters:
    sigma_carbon = (2.5e5 + 3.3e2)*0.5  # Average conductivity of carbon [S/m]***
    sigma_LFP = 2.2e-7  # Electrical conductivity of LiFePO4 [S/m]*
    C_dl_ca = 1.5e-2    # Double-layer capacitance [F/m^2]
#    sigma_ca = 7.50     # Bulk cathode electrical conductivity [S/m]
    D_Li_ca = 4.23e-17   # Bulk diffusion coefficient for Li in LFP [m^2/s]****
    #4.23e-16  # Bulk diffusion coefficient for Li in LFP [m^2/s]**
    D_Li_ca_el = np.array([1e-12, 1e-12, 1.517e-10, 1.517e-10])
    
    # *   Improving the rate performance of LFP by Fe-site doping, Wang et al 2005
    # **  Effect of particle size on DC cond, Ea, and D of LFP in LIBs, Satyavani et al 2016. Values tuned to match performance at 2C
    # *** https://www.thoughtco.com/table-of-electrical-resistivity-conductivity-608499
    #****  http://link.springer.com/10.1007/s10853-014-8395-9 Chang, et al., J Mat Sci, "Effects of particle size and carbon coating on electrochemical properties of LiFePO4/C prepared by hydrothermal method"
    #***** Bai & Bazant, Nature Communications, 2014
    
    "Transport inputs, polynomial fit coefficients, etc."
    params = {}
    params['Dk_elyte_o'] = D_Li_elyte
    params['D_Li_a'] = 8.794e-17  # 8.794e-11
    params['D_Li_b'] = -3.972e-13 # -3.972e-10
    params['D_Li_c'] = 4.862e-10
    params['D_Li_d'] = 0.28687e-6
    params['D_Li_e'] = 0.74678e-3
    params['D_Li_f'] = 0.44130
    params['D_Li_g'] = 0.5508
    params['D_Li_h'] = 0.4717e-3
    params['D_Li_i'] = -0.4106e-6
    params['D_Li_j'] = 0.1287e-9
    params['D_Li_k'] = 2.0
    
    # Electrolyte conductivity (S/dm) taken from:
    #   A. Nyman, M. Behm, and G. Lindbergh, Electrochim. Acta, 53, 6356 (2008)
    params['sigma_elyte_a'] = 33.29e-1  # 3.329e-3
    params['sigma_elyte_b'] = -25.1e-1  # -7.9373e-5
    params['sigma_elyte_c'] = 1.297e-1  # 0.1297e-9
    
    # Liquid activity coefficient taken from:
    #   A. Nyman, M. Behm, and G. Lindbergh, Electrochim. Acta, 53, 6356 (2008)
    params['gamma_elyte_a'] = 0.28687  # 2.8687e-7
    params['gamma_elyte_b'] = -0.74678 # 7.4678e-4
    params['gamma_elyte_c'] = 0.44103  # 0.44130
    params['gamma_elyte_d'] = 0.5508
    params['gamma_elyte_e'] = 0.4717   # 4.717e-4
    params['gamma_elyte_f'] = -0.4106  # -4.106e-7
    params['gamma_elyte_g'] = 0.1287   # 1.287e-10
    
    params['divCharge'] = np.array([1., 1., 1., -1.])
    
    # t_Li+ (-) taken from:
    #   A. Nyman, M. Behm, and G. Lindbergh, Electrochim. Acta, 53, 6356 (2008)
    params['t_elyte_a'] = 0.4492
    params['t_elyte_b'] = -4.717  # e-4
    params['t_elyte_c'] = 0.4106  # 4.106e-7
    params['t_elyte_d'] = -0.1287 # -1.287e-10

    # Create path for saving outputs:
    if 'C_rate' in locals():
        save_name = str(int(C_rate))+"C"
    elif 'i_ext' in locals():
        save_name = str(int(i_ext/10.))+"_mAcm2"
        # Initialize C_rate.  It will be calculated later:
        C_rate = 0
    else:
        raise Exception("Please specify  either i_ext or C_rate.")

    save_path = 'Outputs/' + save_folder + '/' + kinetics + '/'
    
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

#if __name__ == "__main__":
#    exec(open("li_ion_battery_p2d_init.py").read())
#    exec(open("li_ion_battery_p2d_model.py").read())
