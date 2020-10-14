# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:27:54 2018

@author: dkorff
"""

import numpy as np
import cantera as ct
import importlib
import csv

import li_ion_battery_p2d_inputs
importlib.reload(li_ion_battery_p2d_inputs)
from li_ion_battery_p2d_inputs import Inputs
#import sys
#print(sys.path)

#import sys
#sys.path.append('C:\\Users\\dkorff\\Research\\BatCan-repo')

#from functions.diffusion_coeffs import elyte_diffusion
def initialize():
    # Import Cantera objects:
    anode_obj = ct.Solution(Inputs.canterafile,Inputs.anode_phase)
    elyte_obj = ct.Solution(Inputs.canterafile,Inputs.elyte_phase)
    cathode_obj = ct.Solution(Inputs.canterafile,Inputs.cathode_phase)
    conductor_obj = ct.Solution(Inputs.canterafile,Inputs.metal_phase)

    if Inputs.kinetics == 'MH':
        anode_surf_obj = ct.Interface(Inputs.canterafile, 'lithium_surf_marcus',
            [anode_obj, elyte_obj, conductor_obj])
        cathode_surf_obj = ct.Interface(Inputs.canterafile, 'cathode_surf_marcus',
            [cathode_obj, elyte_obj, conductor_obj])
    elif Inputs.kinetics == 'BV':
        anode_surf_obj = ct.Interface(Inputs.canterafile, 'lithium_surf_butler_volmer', 
            [anode_obj, elyte_obj, conductor_obj])
        cathode_surf_obj = ct.Interface(Inputs.canterafile, 
            'cathode_surf_butler_volmer', [cathode_obj, elyte_obj, conductor_obj])
    elif Inputs.kinetics == 'MHC':
        anode_surf_obj = ct.Interface(Inputs.canterafile, 'lithium_surf_mhc',
            [anode_obj, elyte_obj, conductor_obj])
        cathode_surf_obj = ct.Interface(Inputs.canterafile, 'cathode_surf_mhc',
            [cathode_obj, elyte_obj, conductor_obj])

    # Set Cantera object states 
    anode_obj.TP = Inputs.T, ct.one_atm
    elyte_obj.TP = Inputs.T, ct.one_atm
    anode_surf_obj.TP = Inputs.T, ct.one_atm

    X_an_0 = anode_obj.X

    if hasattr(Inputs, 'X_elyte_init'):
        elyte_obj.X = X_elyte_init

    # Initial conditions: UPDATE TO GENERALIZE FOR MULT. SPECIES - DK 8/31/18
    LiCoO2_0 = Inputs.Li_ca_max - Inputs.SOC_0*(Inputs.Li_ca_max - Inputs.Li_ca_min)
    CoO2_0 = 1 - LiCoO2_0
    X_ca_init = '{}:{}, {}:{}'.format(Inputs.Li_species_cathode, LiCoO2_0,
            Inputs.Vac_species_cathode, CoO2_0)

    cathode_obj.X = X_ca_init
    cathode_obj.TP = Inputs.T, ct.one_atm
    cathode_surf_obj.TP = Inputs.T, ct.one_atm

    X_ca_0 = cathode_obj.X
    X_elyte_0 = elyte_obj.X


    """========================================================================="""
    """========================================================================="""
    """========================================================================="""  
    class anode():

        # Set flag so solver knows whether to implement the anode:
        flag = Inputs.flag_anode

        # Number of nodes in the y-direction:
        npoints = Inputs.npoints_anode

        # Initial conditions: UPDATE TO GENERALIZE FOR MULT. SPEC - DK 8/31/18

        # Anode variables for a given volume include X_Li for each shell, Phi_an,
        #       Phi_elyte, and rho_k_elyte for all elyte species.
        nVars = 2 + elyte_obj.n_species

        # Pointers
        ptr = {}
        ptr['iFar'] = elyte_obj.species_index(Inputs.Li_species_elyte)
        ptr['X_k_elyte'] = np.arange(0,elyte_obj.n_species)
        ptr['Phi_ed'] = elyte_obj.n_species
        ptr['Phi_dl'] = elyte_obj.n_species + 1
        
        ptr_vec = {}
        ptr_vec['X_k_elyte'] = ptr['X_k_elyte']
        ptr_vec['Phi_ed'] = ptr['Phi_ed']
        ptr_vec['Phi_dl'] = ptr['Phi_dl']
        for i in np.arange(1, npoints):
            ptr_vec['X_k_elyte'] = np.append(ptr_vec['X_k_elyte'], 
                                            ptr['X_k_elyte'] + i*nVars)
            ptr_vec['Phi_ed'] = np.append(ptr_vec['Phi_ed'], ptr['Phi_ed'] + i*nVars)
            ptr_vec['Phi_dl'] = np.append(ptr_vec['Phi_dl'], ptr['Phi_dl'] + i*nVars)

        # Lithium/electrolyte reaction interface area per planar electrode area.
        #   Default value is 1 and it will be adjusted based on surface 
        #   roughness to scale the reaction interface area [m^2_rxn_int/
        #   m^2_electrode]
        A_surf = Inputs.anode_roughness

        # Size of solution vector
        nSV = npoints*nVars

        # Array of offsets to point to each node's variables:
        offsets = np.arange(0,int(nSV),int(nVars))

        # Store parameters as class object attributes:
        T = Inputs.T
        C_dl = Inputs.C_dl_an
        X_Li_max = Inputs.Li_an_max
        X_Li_min = Inputs.Li_an_min
        D_Li_ed = Inputs.D_Li_an
        
        if Inputs.anode_SEI_flag:
            R_SEI = Inputs.R_SEI

        # Geometric parameters:
        eps_elyte = Inputs.eps_elyte_sep
        tau_elyte = Inputs.tau_sep
        dyInv = npoints/Inputs.H_an
        dy_el = Inputs.H_an - Inputs.H_Li
        dyInv_el = 1/dy_el

        # Calculate the current density [A/m^2] corresponding to a C_rate of 1:
        oneC = anode_obj.density_mole*Inputs.H_an*ct.faraday/3600

        # Electronic conductivity of the electrode phase:
        sigma_eff_ed = Inputs.sigma_an*eps_elyte/tau_elyte**2
        # Species mobilities and diffusion coefficients:
        u_Li_elyte = (Inputs.D_Li_an_el*eps_elyte/ct.gas_constant
            /Inputs.T/tau_elyte**2)
        
        D_el_eff = Inputs.D_Li_an_el*eps_elyte/tau_elyte**2

        def get_tflag():
            return anode.t_flag
        
        def set_tflag(value):
            anode.t_flag = value    


        """========================================================================="""
        """========================================================================="""
        """========================================================================="""

    class separator():
        # Set a flag to let the solver know whether to implement this class:
        flag = Inputs.flag_sep

        # Number of nodes in the y-direction
        npoints = Inputs.npoints_elyte

        # Number of variables per node:
        nVars = 1 + elyte_obj.n_species

        H = Inputs.H_elyte  # Separator thickness [m]

        tau_sep = Inputs.tau_sep  # Tortuosity of separator

        # Geometric parameters:
        eps = 1 - Inputs.eps_elyte_sep
        eps_elyte = Inputs.eps_elyte_sep
        dyInv = npoints/H
        dy = H/npoints
        tau_sep = tau_sep
        
        # Species mobilities and diffusion coefficients:
        u_Li_elyte = (Inputs.D_Li_elyte*eps_elyte/ct.gas_constant
                    /Inputs.T/tau_sep**2)
        D_el_eff = Inputs.D_Li_elyte*eps_elyte/tau_sep**2

        ptr = {}
        ptr['X_k_elyte'] = np.arange(0,elyte_obj.n_species)
        ptr['Phi'] = elyte_obj.n_species
        
        ptr_vec = {}
        ptr_vec['X_k_elyte'] = anode.nSV + ptr['X_k_elyte']
        ptr_vec['Phi'] = anode.nSV + ptr['Phi']
        
        for i in np.arange(1, npoints):
            ptr_vec['X_k_elyte'] = np.append(ptr_vec['X_k_elyte'], anode.nSV + ptr['X_k_elyte'] + i*nVars)
            ptr_vec['Phi'] = np.append(ptr_vec['Phi'], anode.nSV + ptr['Phi'] + i*nVars)

        # Size of the solution vector
        nSV = npoints*nVars

        # Array of offsets to point to each node's variable:
        offsets = np.arange(int(anode.nSV),int(anode.nSV)+int(nSV),int(nVars))


    """========================================================================="""
    """========================================================================="""
    """========================================================================="""

    class cathode():
        # Set flag so solver knows it's looking at the anode class
        flag = Inputs.flag_cathode

        # Number of nodes in the y-direction:
        npoints = Inputs.npoints_cathode

        # Number of "shells" in the cathode particle
        nshells = Inputs.n_shells_cathode

        # Number of state variables per node:
        nVars = nshells + 2 + elyte_obj.n_species

        # Pointers
        ptr = {}
        ptr['iFar'] = elyte_obj.species_index(Inputs.Li_species_elyte)
        ptr['X_ed'] = np.arange(0, nshells)
        ptr['X_k_elyte'] = nshells + np.arange(0,elyte_obj.n_species)
        ptr['Phi_ed'] = nshells + elyte_obj.n_species
        ptr['Phi_dl'] = nshells + elyte_obj.n_species + 1
        
        ptr_vec = {}
        
        ptr_vec['X_ed'] = anode.nSV + separator.nSV + ptr['X_ed']
        ptr_vec['X_k_elyte'] = anode.nSV + separator.nSV + ptr['X_k_elyte']
        ptr_vec['Phi_ed'] = anode.nSV + separator.nSV + ptr['Phi_ed']
        ptr_vec['Phi_dl'] = anode.nSV + separator.nSV + ptr['Phi_dl']
        for i in np.arange(1, npoints):
            ptr_vec['X_ed'] = np.append(ptr_vec['X_ed'], anode.nSV + separator.nSV + ptr['X_ed'] + i*nVars)
            ptr_vec['X_k_elyte'] = np.append(ptr_vec['X_k_elyte'], 
                                            anode.nSV + separator.nSV + ptr['X_k_elyte'] + i*nVars)
            ptr_vec['Phi_ed'] = np.append(ptr_vec['Phi_ed'], anode.nSV + separator.nSV + ptr['Phi_ed'] + i*nVars)
            ptr_vec['Phi_dl'] = np.append(ptr_vec['Phi_dl'],anode.nSV + separator.nSV +  ptr['Phi_dl'] + i*nVars)

        # Cathode/elyte interface area per unit volume
        #   [m^2 interface / m_3 total electrode volume]
        # For spherical particles, the total surface area per unit volume can be
        #   calculated from the geometry. Since some particles will overlap, we
        #   multiply by an 'overlap' input parameter.
        A_surf = (1-Inputs.overlap_ca)*6*Inputs.eps_solid_ca/Inputs.d_part_ca

        # Set up the solution vector
        nSV = npoints*nVars

        # Array of offsets to point to each node's variable:
        offsets = np.arange(int(anode.nSV + separator.nSV), \
                        int(anode.nSV + separator.nSV)+int(nSV),int(nVars))

        #  Store Parameters in the Class object:
        T = Inputs.T
        C_dl = Inputs.C_dl_ca
        X_Li_max = Inputs.Li_ca_max
        X_Li_min = Inputs.Li_ca_min
        D_Li_ed = Inputs.D_Li_ca

        # Geometric parameters:
        H = Inputs.H_ca
        eps_ed = Inputs.eps_solid_ca
        eps_elyte = 1 - Inputs.eps_solid_ca
        tau_ed = Inputs.tau_ca
        r_p = Inputs.r_p_ca
        d_part = Inputs.d_part_ca
        dyInv = npoints/Inputs.H_ca
        dy = Inputs.H_ca/npoints
        dr = d_part*0.5/nshells
        rho_ed = 1/((Inputs.wt_pct_active/cathode_obj.density_mass) + \
                (Inputs.wt_pct_cond/Inputs.conductor_rho) + \
                (Inputs.wt_pct_bind/Inputs.binder_rho))

        mass = H*rho_ed*eps_ed

        eps_active = rho_ed*Inputs.wt_pct_active/cathode_obj.density_mass

        # Calculate the current density [A/m^2] corresponding to a C_rate of 1:
        oneC = eps_ed*cathode_obj.density_mole*Inputs.H_ca*ct.faraday/3600

        # Calculate the percent volume of a single particle that exists in
        #   each 'shell'. I.e. for shell j, what is the volume of that shell,
        #   relative to the total particle volume? The radius of the volume is
        #   currently discretized evenly (i.e. 'dr' the differential radius is
        #   constant). Certainly other discretizations (such as constant
        #   differential volume) are possible (and perhaps better).
        #
        #   Because the volume is 4/3 pi*r^3, the volume of the shell relative 
        #   to the total volume is (r_shell/r_particle)^3, and the differential 
        #   volume relative to the total, for shell 'j' is:
        #       (r_shell(j+1)^3 - r_shell(j)^3)/r_particle^3
        #   Because the radius is discretized evenly, the radius of shell j, 
        #      r_j, relative to the total radius r_particle, is:
        #       r_j/r_particle = j/nshells

        V_shell = np.zeros([nshells])
        nsr3 = 1/nshells/nshells/nshells
        for j in np.arange(0, nshells, 1):
            V_shell[j] = ((j+1)**3 - (j)**3)*nsr3

        sigma_eff_solid = ((Inputs.wt_pct_active*Inputs.sigma_LFP 
                            + Inputs.wt_pct_cond*Inputs.sigma_carbon)
                            / (Inputs.wt_pct_active+Inputs.wt_pct_cond))
        sigma_eff_ed = sigma_eff_solid*eps_ed/tau_ed**2
        
        u_Li_elyte = (Inputs.D_Li_ca_el*eps_elyte/ct.gas_constant
            /Inputs.T/tau_ed**2)
        
        D_el_eff = Inputs.D_Li_ca_el*eps_elyte/tau_ed**2

        def get_tflag():
            return cathode.t_flag
        
        def set_tflag(value):
            cathode.t_flag = value  

    """========================================================================="""
    """========================================================================="""
    """========================================================================="""
    class battery:
        
        # The pointer dictionary created here depends on the structure of the 
        #   electrolyte object in the CTI file. The default assumption is that
        #   the organic solvents are the first x number of entries, the lithium
        #   species is second to last, and the anion species is last
        ptr_el = {}
        ptr_el['solvents'] = np.arange(0, elyte_obj.n_species-2)
        ptr_el['Li'] = elyte_obj.n_species - 2
        ptr_el['PF6'] = elyte_obj.n_species - 1
        
        cst_params = Inputs.params.copy()
        
        H = Inputs.H_an + Inputs.H_ca + Inputs.H_elyte
        
        eps = (Inputs.H_an+cathode.eps_ed*Inputs.H_ca+ \
            separator.eps*Inputs.H_elyte)/H
            
        rho = (anode_obj.density_mass*Inputs.H_an + \
            Inputs.rho_sep*separator.eps*Inputs.H_elyte + \
            cathode.rho_ed*cathode.eps_ed*cathode.H)/H
        
        mass = (Inputs.H_an*anode_obj.density_mass + 
                Inputs.H_ca*cathode.eps_ed*cathode_obj.density_mass +
                Inputs.H_elyte*separator.eps*Inputs.rho_sep)
        
    """========================================================================="""
    """========================================================================="""
    """========================================================================="""

    class current():        
        def get_i_ext():
            return current.i_ext
        
        def set_i_ext(value):
            current.i_ext = value
            
        # Calculate the actual current density:
        # The minus sign is because we begin with the charging reaction, which
        #   delivers negative charge to the anode:
        if hasattr(Inputs,'i_ext'):
            i_ext_set = -Inputs.i_ext 
            i_ext_amp = -Inputs.i_ext
        else:
            i_ext_set = -Inputs.C_rate*min(anode.oneC, cathode.oneC)  
            i_ext_amp = -Inputs.C_rate*min(anode.oneC, cathode.oneC)
    class solver_inputs():     
        
        dG_anode = anode_surf_obj.delta_gibbs[0]
        dG_cathode = cathode_surf_obj.delta_gibbs[0]

        # Note that this assumes a single-electron transfer reaction, and that 
        #   there is but a single reaction:
        phi_dl_an = -dG_anode/ct.faraday
        phi_dl_ca = -dG_cathode/ct.faraday

        # Initialize the solution vector:
        SV_0 = np.zeros([anode.nSV+separator.nSV+cathode.nSV])
        
        # Set up algebraic variable vector:
        algvar = np.zeros_like(SV_0)
        
        offsets = anode.offsets
        ptr = anode.ptr
        for j in range(anode.npoints):   
            # Electrolyte species mole fraction in the anode: 
            SV_0[offsets[j] + ptr['X_k_elyte']] = elyte_obj.X
            # This is a differential variable:
            algvar[offsets[j] + ptr['X_k_elyte']] = 1
        
            # Electrode phase electric potential (an algebraic variable):
            SV_0[offsets[j] + ptr['Phi_ed']] = Inputs.Phi_anode_init
            # Anode/electrolyte double layer potential:
            SV_0[offsets[j] + ptr['Phi_dl']] = \
                Inputs.Phi_elyte_init - Inputs.Phi_anode_init
            # This is a differential variable:
            algvar[offsets[j] + ptr['Phi_dl']] = 1
        
        
        offsets = separator.offsets
        ptr = separator.ptr
        for j in np.arange(0, separator.npoints):
            # Electrolyte species mole fraction in the separator:
            SV_0[offsets[j] + ptr['X_k_elyte']] = elyte_obj.X
            # This is a differential variable:
            algvar[offsets[j] + ptr['X_k_elyte']] = 1
        
            # Electrolyte electrice potential (an algebraic variable):
            SV_0[offsets[j] + ptr['Phi']] = Inputs.Phi_elyte_init
        
        offsets = cathode.offsets
        ptr = cathode.ptr
        for j in range(cathode.npoints):
            # Cathode phase lithium mole fraction:
            SV_0[offsets[j] + ptr['X_ed']] = \
                np.ones([cathode.nshells])*X_ca_0[0]
            # This is a differential variable:
            algvar[offsets[j] + ptr['X_ed']] = 1
        
            # Electrolyte species mole fractions in the cathode:
            SV_0[offsets[j] + ptr['X_k_elyte']] = elyte_obj.X
            # This is a differential variable:
            algvar[offsets[j] + ptr['X_k_elyte']] = 1
        
            # Cathode phase electric potential (an algebraic variable):
            SV_0[offsets[j] + ptr['Phi_ed']] = \
                Inputs.Phi_anode_init + Inputs.Delta_Phi_init
        
            # Cathode/electrolyte double layer potential:
            SV_0[offsets[j] + ptr['Phi_dl']] = \
                Inputs.Phi_elyte_init - (Inputs.Phi_anode_init + Inputs.Delta_Phi_init)
            # This is a differential variable:
            algvar[offsets[j] + ptr['Phi_dl']] = 1
    
    # Dict of params:
    params = {'bat_rho': battery.rho, 'bat_H': battery.H, 
            'bat_eps': battery.eps, 'ca_rho': cathode.rho_ed, 
            'ca_H': cathode.H, 'ca_eps': cathode.eps_ed}

    # Write parameters to 'test.csv' file:
    with open('test.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(params.keys())
        w.writerow(params.values())

    return (anode, cathode, separator, solver_inputs, current, battery, 
        anode_obj, anode_surf_obj, elyte_obj, cathode_obj, cathode_surf_obj, conductor_obj)