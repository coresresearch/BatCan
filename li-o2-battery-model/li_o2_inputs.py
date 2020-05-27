"========== USER INPUTS for 1D Li-O2 Battery simulation =========="

" Battery operation and simulation parameters "
# Simulation time and external current
# Equilibrate at zero current for this amount of time, before modeling battery
#   operation:
t_ocv = 1e8                                     # [s]
# Discharge or charge current.
# TODO #5: calculate i_ext as a function of a user-supplied C-rate:
i_ext = -1.5e-1                       # External current [A/m2]
DoD = 0.75                          # Depth of discharge (out of 1.0)
# TODO #6: calculate tspan as a function of C_rate and DoD
tspan = DoD*23e7*9e-3/abs(i_ext)    # Simulation time for charge/discharge [s]

phi_elyte_init = -3.19              # double layer voltage [V]

Molarity_elyte = 1.0                # Moles of Li salt per L of electrolyte.

TP = 300, 21278.25                    # inital temp, pressure [K, Pa]

# Tolerances (absolute and relative)
atol = 1e-6
rtol = 1e-3

" Microstructure and geometry "
# Discretization
N_y_ca = 10                         # no. cells in cathode
N_y_sep = 1                          # no. cells in separator

# Geometric/microstructure parameters
th_cath = 600e-6                     # cathode thickness [m]
th_sep = 1.55e-3                    # separator thickness [m]
d_part = 10e-6                      # carbon particle diameter [m]
d_oxide = 2e-6                      # oxide particle diameter [m]
th_oxide = 0.5e-6                   # thickness of oxide ellipsoid [m]

# Porosoties
eps_sep = 0.9                         # separator porosity
eps_elyte_init = 0.5                # initial electrolyte volume fraction [-]
eps_oxide_init = 1e-12              # initial oxide volume fraction [-]
eps_binder_init = 0.                # initial binder volume fraction [-]

"Transport"
# Select transport model:
#  - cst (concentrated solution theory) or 
#  - dst (dilute solution theory)
transport = 'cst'                   

# Bruggeman coefficient for relating the tortuosity factor to the porosity:
n_bruggeman = 1.5

# Properties
C_dl = 1.1e-6                  # double layer capacitance [F/m2]

# Electrolyte bulk diffusion coefficients.
#    Dict keys must match species names in Cantera input file.
D_k_elyte = {}
D_k_elyte['Li+(e)'] = 4e-10          # bulk diff coeff Li+ in elyte (m2/s)
D_k_elyte['PF6-(e)'] = 4e-10         # bulk diff coeff PF6- in elyte (m2/s)
D_k_elyte['O2(e)'] = 7e-10           # bulk diff coeff O2 in elyte (m2/s)
D_k_elyte['C3H4O3(e)'] = 1.          # EC diffusion is fast
D_k_elyte['C4H8O3(e)'] = 1.          # EMC diffusion is fast

# Cantera's name for the electrolyte species assigned to double-layer currents 
# (most likely the Li ion).
dl_species = 'Li+(e)'

# For transport calculations, need the names given to the Li ion, the 
#    counter-ion, and the solvent molecules. 
Li_elyte_name = 'Li+(e)'
counter_ion_elyte_name = 'PF6-(e)'
solvent_elyte_names = 'C3H4O3(e)', 'C4H8O3(e)'

# CST Parameters:
D_Li_CST = 8.794e-17, -3.972e-13, 4.862e-10, 0.28687e-6, 0.74678e-3, 0.44130, \
    0.5508, 0.4717e-3, -0.4106e-6, 0.1287e-9, 2.0

# Electrolyte conductivity (S/m)
sigma_elyte_CST = 3.329e-3, -7.9373e-5, 0.1297e-9

# Liquid activity coefficient
gamma_elyte_CST = 2.8687e-7, 7.4678e-4, 0.44130, 0.5508, 4.717e-4, -4.106e-7, \
    1.287e-10

t_elyte_CST = 0.4492, -4.717e-4, 4.106e-7, -1.287e-10

ctifile = 'Li_Air.cti'