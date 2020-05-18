" Initialize variables, parameters, and data structures for simulation"

" Load necessary modules "
"============================================================================"
import numpy as np
import cantera as ct

" Load in user inputs "
"============================================================================"
from li_o2_inputs import *

"============================================================================"
# Import necessary phases
gas = ct.Solution(ctifile,'gas')
cath_b = ct.Solution(ctifile,'graphite')
elyte = ct.Solution(ctifile,'electrolyte')
oxide = ct.Solution(ctifile,'Li2O2')
tpb = ct.Interface(ctifile,'cathode_tpb',[elyte, oxide, cath_b])
inter = ct.Interface(ctifile,'cathode_surf',[elyte,cath_b])
air_elyte = ct.Interface(ctifile,'air_elyte',[gas,elyte])
Li_b = ct.Solution(ctifile,'Lithium')
Li_s = ct.Interface(ctifile,'Li_surface',[Li_b,elyte])

# Set phase temperatures
gas.TP = TP
air_elyte.TP = TP
elyte.TP = TP
oxide.TP = TP
inter.TP = TP
tpb.TP = TP
cath_b.TP = TP
Li_b.TP = TP
Li_s.TP = TP

# Store these phases in a common 'objs' dict
objs = {}
objs['gas'] = gas
objs['cath_b'] = cath_b
objs['elyte'] = elyte
objs['oxide'] = oxide
objs['inter'] = inter
objs['tpb'] = tpb
objs['air_elyte'] = air_elyte
objs['Li_b'] = Li_b
objs['Li_s'] = Li_s

# Microstructure and geometry:
# initial carbon volume fraction [-]
eps_carbon = 1. - eps_elyte_init - eps_binder_init - eps_oxide_init      
# Volume of a single carbon particle [m3]
V_part = 4/3 * np.pi * (d_part / 2)**3
# Surface area of a single carbon particle [m2]
A_part = 4 * np.pi * (d_part / 2)**2
# Carbon innterface area per unit volume cathode [m2/m3 total]
A_int = eps_carbon * A_part / V_part
# Oxide area contacting carbon particle, per particle [m2]
A_oxide = np.pi * d_oxide**2 / 4
# Volume of a single oxide torroid [m3]
V_oxide = 2/3 * np.pi * (d_oxide/2)**2 * th_oxide   

# Inverse thickness of a single cathode volume discritization [1/m]
dyInv_cath = Ny_cath / th_cath  

# inverse thickness of a single separator volume discritization [1/m]     
dyInv_sep = Ny_sep / th_sep                         

geom = {}
geom['bruggman'] = n_bruggeman
geom['tau_cath'] = eps_carbon / (eps_carbon**(1+geom['bruggman']))

# Store pointers in a common 'ptr' dict
ptr = {}
# electron in the inter net_production_rates vector
ptr['elec'] = elyte.n_species
# electrolyte in the inter net_production_rates vector
ptr['elyte'] = np.arange(0,elyte.n_species)
# location of O2 in elyte phase
ptr['O2_elyte'] = 0
# location of Li+ in elyte phase
ptr['Li+'] = 1
# location of PF6- in elyte phase
ptr['PF6-'] = 2
# location of EC in elyte phase
ptr['EC'] = 3                  
# location of EMC in elyte phase  
ptr['EMC'] = 4                 

# Store parameters in a common 'params' dict
params = {}
params['i_ext'] = i_ext
params['T'] = TP[0]
params['E_elyte_0'] = eps_elyte_init
# params['sigma'] = sigma_io
params['rtol'] = rtol
params['atol'] = atol
params['Ny_cath'] = Ny_cath
params['dyInv_cath'] = dyInv_cath
params['Ny_sep'] = Ny_sep
params['dyInv_sep'] = dyInv_sep
params['A_int'] = A_int
params['C_dl'] = C_dl
params['E_carb'] = eps_carbon
params['E_sep_inv'] = 1/eps_sep
params['transport'] = transport
params['r_oxide'] = 0.5*d_oxide
params['molar_vol oxide'] = 45.881*.001 / 2.31 # Molar volume of Li2O2 [m3/kmol]

# CST Parameters:
params['D_Li_a'] = 8.794e-17
params['D_Li_b'] = -3.972e-13
params['D_Li_c'] = 4.862e-10
params['D_Li_d'] = 0.28687e-6
params['D_Li_e'] = 0.74678e-3
params['D_Li_f'] = 0.44130
params['D_Li_g'] = 0.5508
params['D_Li_h'] = 0.4717e-3
params['D_Li_i'] = -0.4106e-6
params['D_Li_j'] = 0.1287e-9
params['D_Li_k'] = 2.0

# Electrolyte conductivity (S/m)
params['sigma_elyte_a'] = 3.329e-3
params['sigma_elyte_b'] = -7.9373e-5
params['sigma_elyte_c'] = 0.1297e-9

# Liquid activity coefficient
params['gamma_elyte_a'] = 2.8687e-7
params['gamma_elyte_b'] = 7.4678e-4
params['gamma_elyte_c'] = 0.44130
params['gamma_elyte_d'] = 0.5508
params['gamma_elyte_e'] = 4.717e-4
params['gamma_elyte_f'] = -4.106e-7
params['gamma_elyte_g'] = 1.287e-10

params['divCharge'] = 1

params['t_elyte_a'] = 0.4492
params['t_elyte_b'] = -4.717e-4
params['t_elyte_c'] = 4.106e-7
params['t_elyte_d'] = -1.287e-10

params['iO2_elyte'] = 0
params['iLi_elyte'] = 1
params['iPF6_elyte'] = 2
params['iEC_elyte'] = 3
params['iEMC_elyte'] = 4

params['Dk_elyte_o'] = np.zeros_like(elyte.X)
params['Zk_elyte'] = np.zeros_like(elyte.X)
for j in elyte.species_names:
    params['Dk_elyte_o'][elyte.species_index(j)] = D_k_elyte[j]
    params['Zk_elyte'][elyte.species_index(j)] = elyte.species(j).charge

params['polarity_k'] = np.sign(params['Zk_elyte'])
# Store elyte diffusion coefficients
# params['Dk_elyte_o'] = np.zeros(elyte.n_species)
# params['Dk_mig_elyte_o'] = np.zeros(elyte.n_species)

# params['Dk_elyte_o'][ptr['PF6-']] = D_PF6_elyte / geom['tau_cath']**3
# params['Dk_elyte_o'][ptr['Li+']] = D_Li_elyte / geom['tau_cath']**3
# params['Dk_elyte_o'][ptr['O2_elyte']] = D_O2_elyte / geom['tau_cath']**3
# params['Dk_elyte_o'][ptr['EC']] = D_EC_elyte / geom['tau_cath']**3
# params['Dk_elyte_o'][ptr['EMC']] = D_EMC_elyte / geom['tau_cath']**3

params['Dk_mig_elyte_o'] = params['Dk_elyte_o']*params['Zk_elyte']*ct.faraday\
     / ct.gas_constant / params['T']


# Store solution vector pointers in a common 'SVptr' dict
SVptr = {}
SVptr['phi'] = 0                # double layer potential in SV
SVptr['elyte'] = np.arange(1, elyte.n_species + 1)
SVptr['E_oxide'] = SVptr['elyte'][-1]+1                                    # cathode electrolyte concentrations in SV
#SVptr['theta'] = np.arange(SVptr['elyte'][-1]+1,SVptr['elyte'][-1]+1 + len(inter.X))    # surface coverage in SV
#SVptr['sep_phi'] = (SVptr['theta'][-1]+1)*Ny_cath                                       # separator double layer potential in SV
#SVptr['sep_elyte'] = np.arange((SVptr['theta'][-1]+1)*Ny_cath , (SVptr['theta'][-1]+1)*Ny_cath +elyte.n_species)  # separator electrolyte concentrations in SV

# Store plot pointers in a common 'pltptr' dict
pltptr = {}
pltptr['O2'] = elyte.species_index('O2(e)')
pltptr['Li+'] = elyte.species_index('Li+(e)')
pltptr['PF6-'] = elyte.species_index('Pf6-(e)')
pltptr['EC'] = elyte.species_index('C3H4O3(e)')
pltptr['EMC'] = elyte.species_index('C4H8O3(e)')

# Set inital values
rho_elyte_init = elyte.Y*elyte.density                              # electrolyte concentrations
#theta_init = [theta_init, 1-theta_init]                                                 # surface coverages

SV_single_cath = np.r_[phi_elyte_init,rho_elyte_init,eps_oxide_init]    # store in an array
SV0_cath = np.tile(SV_single_cath,Ny_cath)                          # tile for discritization
params['SV_single_cath'] = len(SV_single_cath)                      # put length of single cathode SV into 'params' for indexing

#SV_single_sep = rho_elyte_init
#SV_elyte = np.tile(SV_single_sep,Ny_sep)
#SV0_sep = np.r_[SV_elyte,phi_elyte_init]
rho_elyte_sep_init = rho_elyte_init
SV_single_sep = np.r_[rho_elyte_sep_init]                         # electric potential and species mass densities in separator
SV0_sep = np.tile(SV_single_sep,Ny_sep)                             # tile for discritization
params['SV_single_sep'] = len(SV_single_sep)                        # put length of single separator SV into 'params' for indexing

#SV0 = np.r_[SV0_cath,SV0_sep]                                       # combine initial values
SV0 = SV0_cath