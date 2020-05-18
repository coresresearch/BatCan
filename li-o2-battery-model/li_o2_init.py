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
cath_b = ct.Solution(ctifile,'graphite')
elyte = ct.Solution(ctifile,'electrolyte')
oxide = ct.Solution(ctifile,'Li2O2')
tpb = ct.Interface(ctifile,'cathode_tpb',[elyte, oxide, cath_b])
inter = ct.Interface(ctifile,'cathode_surf',[elyte,cath_b])
air_elyte = ct.Interface(ctifile,'air_elyte',[gas,elyte])
Li_b = ct.Solution(ctifile,'Lithium')
Li_s = ct.Interface(ctifile,'Li_surface',[Li_b,elyte])

# Set temperature and pressure for all phse objects:
gas.TP = TP
air_elyte.TP = TP
elyte.TP = TP
oxide.TP = TP
inter.TP = TP
tpb.TP = TP
cath_b.TP = TP
Li_b.TP = TP
Li_s.TP = TP

# Store these phases in a common 'objs' dict.
# This takes the data from Cantera and stores data for use in a Python 
# dictionary so we can efficiently pass them to various functions. A dictionary
# associates a key word with a value.
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

"============================================================================"
"  INPUT PARAMETER STORAGE  "
# Store parameters in a common 'params' dict
params = {}
params['i_ext'] = i_ext
params['TP'] = TP
params['rtol'] = rtol
params['atol'] = atol
params['Ny_cath'] = Ny_cath
# Inverse thickness of a single cathode volume discritization [1/m]
params['dyInv_cath'] =  Ny_cath / th_cath
params['Ny_sep'] = Ny_sep
# inverse thickness of a single separator volume discritization [1/m] 
params['dyInv_sep'] =  Ny_sep / th_sep
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

params['E_sep_inv'] = 1/eps_sep

"============================================================================"
"  TRANSPORT PEROPERTIES -- CALCULATION AND STORAGE  "
# Transport model: dilute (dst) or concentrated (cst) solution theory:
params['transport'] = transport

# Bruggeman coefficient
params['bruggman'] = n_bruggeman
# Initial cathode tortuosity factor:
# TODO: calculate this property in real-time.
params['tau_cath'] = params['eps_carbon']**(-params['bruggman'])

# Store elyte diffusion coefficients and elementary charge:
params['Dk_elyte_o'] = np.zeros_like(elyte.X)
params['Zk_elyte'] = np.zeros_like(elyte.X)
for j in elyte.species_names:
    params['Dk_elyte_o'][elyte.species_index(j)] = D_k_elyte[j]
    params['Zk_elyte'][elyte.species_index(j)] = elyte.species(j).charge

params['polarity_k'] = np.sign(params['Zk_elyte'])
params['Dk_mig_elyte_o'] = params['Dk_elyte_o']*params['Zk_elyte']*ct.faraday\
     / ct.gas_constant / params['TP'][0]

if params['transport'] == 'cst':
    # Cantera species indices for relevant species:
    params['i_Li_elyte'] = elyte.species_index(Li_elyte_name)
    params['i_counter_ion'] = elyte.species_index(counter_ion_elyte_name)
    params['i_solvent_elyte'] = []
    for n_sp in solvent_elyte_names:
        params['i_solvent_elyte'].append(elyte.species_index(n_sp))

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
    nz_charges = abs(params['Zk_elyte'])
    # Now overwrite all zeros with ones.
    nz_charges[nz_charges == 0] = 1.
    # Divide by this new array. For un-cahrged species, the transferrence number is already equal to zero, so the value of `div_charge` is irrelevant:
    params['div_charge'] = 1./nz_charges

"============================================================================"
"  SOLUTION VECTORE SETUP  "
# Store solution vector pointers in a common 'SVptr' dict
SVptr = {}
# Variables per finite volume (aka 'node'): elyte electric potential, oxide 
#     density, electrolyte species densities
nvars_node = int(elyte.n_species + 2)

SVptr['phi'] = 0                # double layer potential in SV
SVptr['elyte'] = np.arange(1, elyte.n_species + 1)
SVptr['E_oxide'] = SVptr['elyte'][-1]+1 
#SVptr['sep_phi'] = (SVptr['theta'][-1]+1)*Ny_cath                                       # separator double layer potential in SV
#SVptr['sep_elyte'] = np.arange((SVptr['theta'][-1]+1)*Ny_cath , (SVptr['theta'][-1]+1)*Ny_cath +elyte.n_species)  # separator electrolyte concentrations in SV

# Set inital values
rho_elyte_init = elyte.Y*elyte.density     # electrolyte concentrations

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

"============================================================================"
"  POINTER FOR POST-PROCESSING / PLOTTING  "
# Store plot pointers in a common 'pltptr' dict
pltptr = {}
pltptr['O2'] = elyte.species_index('O2(e)')
pltptr['Li+'] = elyte.species_index('Li+(e)')
pltptr['PF6-'] = elyte.species_index('Pf6-(e)')
pltptr['EC'] = elyte.species_index('C3H4O3(e)')
pltptr['EMC'] = elyte.species_index('C4H8O3(e)')