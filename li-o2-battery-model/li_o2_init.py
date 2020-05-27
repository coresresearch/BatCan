" Initialize variables, parameters, and data structures for simulation"

" Load necessary modules "
"============================================================================"
import numpy as np
import cantera as ct

" Load in user inputs "
"============================================================================"
from li_o2_inputs import *

"============================================================================"
"  CANTERA OJECT SETUP  "
# Import necessary phases from Cantera
# Make objects to handle calculations
gas = ct.Solution(ctifile,'gas')
ca_bulk = ct.Solution(ctifile,'graphite')
elyte = ct.Solution(ctifile,'electrolyte')
oxide = ct.Solution(ctifile,'Li2O2')
tpb = ct.Interface(ctifile,'cathode_tpb',[elyte, oxide, ca_bulk])
ca_surf = ct.Interface(ctifile,'cathode_surf',[elyte, ca_bulk])
air_elyte = ct.Interface(ctifile,'air_elyte',[gas, elyte])
Li_bulk = ct.Solution(ctifile,'Lithium')
Li_surf = ct.Interface(ctifile,'Li_surface',[Li_bulk, elyte])

# Set temperature and pressure for all phse objects:
gas.TP = TP
air_elyte.TP = TP
elyte.TP = TP
oxide.TP = TP
ca_surf.TP = TP
tpb.TP = TP
ca_bulk.TP = TP
Li_bulk.TP = TP
Li_surf.TP = TP

# Store these phases in a common 'objs' dict.
# This takes the data from Cantera and stores data for use in a Python 
# dictionary so we can efficiently pass them to various functions. A dictionary
# associates a key word with a value.
objs = {}
objs['gas'] = gas
objs['ca_bulk'] = ca_bulk
objs['elyte'] = elyte
objs['oxide'] = oxide
objs['ca_surf'] = ca_surf
objs['tpb'] = tpb
objs['air_elyte'] = air_elyte
objs['Li_bulk'] = Li_bulk
objs['Li_surf'] = Li_surf

"============================================================================"
"  INPUT PARAMETER STORAGE  "
# Store parameters in a common 'params' dict
params = {}
params['i_ext'] = i_ext
params['TP'] = TP
params['rtol'] = rtol
params['atol'] = atol
params['N_y_ca'] = N_y_ca
# Inverse thickness of a single cathode volume discritization [1/m]
params['dyInv_ca'] =  N_y_ca / th_cath
params['N_y_sep'] = N_y_sep
# inverse thickness of a single separator volume discritization [1/m] 
params['dyInv_sep'] =  N_y_sep / th_sep
params['C_dl'] = C_dl
# Carbon volume fraction [-]
params['eps_carbon'] = 1. - eps_elyte_init - eps_binder_init - eps_oxide_init

# Volume of a single carbon particle [m3]
V_carbon = 4./3. * np.pi * (d_part / 2.)**3.
# Surface area of a single carbon particle [m2]
A_carbon = 4. * np.pi * (d_part / 2.)**2.
# Initial carbon innterface area per unit volume cathode [m2/m3 total]
params['A_carbon_init'] = params['eps_carbon'] * A_carbon / V_carbon

# Oxide particle radius:
params['r_oxide'] = 0.5*d_oxide

# Index of the electrolyte species responsible for double layer current
#   (this is usually the Li ion).
params['i_dl_species'] = elyte.species_index(dl_species)

params['eps_sep_inv'] = 1./eps_sep

"============================================================================"
"  TRANSPORT PEROPERTIES -- CALCULATION AND STORAGE  "
# Transport model: dilute (dst) or concentrated (cst) solution theory:
params['transport'] = transport

# Bruggeman coefficient
params['bruggman'] = n_bruggeman

# Store elyte diffusion coefficients and elementary charge:
params['D_k_elyte_o'] = np.zeros_like(elyte.X)
params['Z_k_elyte'] = np.zeros_like(elyte.X)
for j in elyte.species_names:
    params['D_k_elyte_o'][elyte.species_index(j)] = D_k_elyte[j]
    params['Z_k_elyte'][elyte.species_index(j)] = elyte.species(j).charge

params['polarity_k'] = np.sign(params['Z_k_elyte'])
params['D_k_mig_elyte_o'] = params['D_k_elyte_o']*params['Z_k_elyte'] \
    * ct.faraday / ct.gas_constant / params['TP'][0]

# Cantera species indices for relevant species:
params['i_Li_elyte'] = elyte.species_index(Li_elyte_name)
params['i_counter_ion'] = elyte.species_index(counter_ion_elyte_name)
params['i_solvent_elyte'] = []
for n_sp in solvent_elyte_names:
    params['i_solvent_elyte'].append(elyte.species_index(n_sp))

if params['transport'] == 'cst':
    # Li diffusion coefficients:
    params['D_Li_CST'] = D_Li_CST
    # Transferrence number coefficients:
    params['t_elyte'] = t_elyte_CST
    # Ionic conductivity coefficients:
    params['sigma_elyte'] = sigma_elyte_CST
    # Li activity coefficients:
    params['gamma_elyte'] = gamma_elyte_CST
    # Magnitude of 1/z_k for electrolyte species.  Need to take care to prevent 
    #  division by zero, so we really just staore the non-zero cahrges 
    #  ('nz_charges'). First create a copy of the electrolyte species charges:
    nz_charges = abs(params['Z_k_elyte'])
    # Now overwrite all zeros with ones.
    nz_charges[nz_charges == 0] = 1.
    # Divide by this new array. For un-cahrged species, the transferrence number is already equal to zero, so the value of `div_charge` is irrelevant:
    params['div_charge'] = 1./nz_charges



# Use the user-provided electrolyte salt molarity to set the initial 
#    electrolyte composition:
# Li+ and PF6- molar masses:
W_Li = elyte.molecular_weights[params['i_Li_elyte']]
Y_Li = W_Li*Molarity_elyte/elyte.density_mass
W_counter = elyte.molecular_weights[params['i_counter_ion']]
Y_counter = W_counter*Molarity_elyte/elyte.density_mass

# Save initial mass fractions from input file.  Assume all non-salt species should keep the same ratio.
Y_initial = elyte.Y

# Initial salt mass fraction from input file:
Y_salt_initial = Y_initial[params['i_Li_elyte']] \
    + Y_initial[params['i_counter_ion']]
# New salt mass fraction:
Y_salt_new = Y_Li + Y_counter
# CHange in salt mass fraction:
dY_salt = Y_salt_new - Y_salt_initial

# Scale all other species mass fractions so that the sum of all mass fractions 
#    equals one:
scaling_factor = 1 - dY_salt/(1 - Y_salt_initial)
Y_new = Y_initial * scaling_factor

# Set salt species mass fractions:
Y_new[params['i_Li_elyte']] = Y_Li
Y_new[params['i_counter_ion']] = Y_counter

# Set Cantera object:
elyte.Y = Y_new

"============================================================================"
"  SOLUTION VECTOR SETUP  "
# Store solution vector pointers in a common 'SVptr' dict
SVptr = {}
# Variables per finite volume (aka 'node'): elyte electric potential, oxide 
#     density, electrolyte species densities
nvars_node_ca = int(elyte.n_species + 2)

# double layer potential in solution vector SV
SVptr['phi_dl'] = range(0, nvars_node_ca*N_y_ca, nvars_node_ca)                
# Oxide volume fraction in solution vector SV
SVptr['eps_oxide'] = range(1,nvars_node_ca*N_y_ca, nvars_node_ca)
# electrolyte species mass densities in solution vector SV
SVptr['C_k_elyte_ca'] = np.ndarray(shape=(N_y_ca, elyte.n_species),\
    dtype='int')
for j in range(N_y_ca):
    SVptr['C_k_elyte_ca'][j,:] = range(2 + j*nvars_node_ca, \
        2 + j*nvars_node_ca + elyte.n_species)

nvars_node_sep = int(elyte.n_species)
SVptr['C_k_elyte_sep'] = np.ndarray(shape=(N_y_sep, elyte.n_species),\
    dtype='int')
for j in range(N_y_sep):
    SVptr['C_k_elyte_sep'][j,:] = range(nvars_node_ca*N_y_ca \
        + j*nvars_node_sep, nvars_node_ca*N_y_ca + j*nvars_node_sep \
        + elyte.n_species)

# Set inital values
# Electrolyte species mass densities (kg per m^3 of electrolyte)
C_k_elyte_init = elyte.concentrations

# Store in an array:
SV_single_cath = np.r_[phi_elyte_init, eps_oxide_init, C_k_elyte_init]
# Tile for discritization
SV0_cath = np.tile(SV_single_cath, N_y_ca)

SV_single_sep = C_k_elyte_init
SV0_sep = np.tile(SV_single_sep, N_y_sep)

# SV0 = np.r_[SV0_cath,SV0_sep]                                       # combine initial values
SV0 = SV0_cath

"============================================================================"
"  POINTER FOR POST-PROCESSING / PLOTTING  "
# Store plot pointers in a common 'pltptr' dict
pltptr = {}
pltptr['O2'] = elyte.species_index('O2(e)')
pltptr['Li+'] = elyte.species_index('Li+(e)')
pltptr['PF6-'] = elyte.species_index('Pf6-(e)')
pltptr['EC'] = elyte.species_index('C3H4O3(e)')
pltptr['EMC'] = elyte.species_index('C4H8O3(e)')