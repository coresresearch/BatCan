"========== USER INPUTS for 1D Li-O2 Battery simulation =========="

" Battery operation and simulation parameters "
# Simulation time and external current
# Equilibrate at zero current for this amount of time, before modeling battery
#   operation:
t_ocv = 1e8                                     # [s]
# Discharge or charge current.
# TODO #5: calculate i_ext as a function of a user-supplied C-rate:
i_ext = -1e-3                       # External current [A/m2]
DoD = 0.25                          # Depth of discharge (out of 1.0)
# TODO #6: calculate tspan as a function of C_rate and DoD
tspan = DoD*23e6*9e-3/abs(i_ext)    # Simulation time for charge/discharge [s]

phi_elyte_init = -3.19              # double layer voltage [V]

TP = 300, 101325                    # inital temp, pressure [K, Pa]

# Tolerances (absolute and relative)
atol = 1e-8
rtol = 1e-6

" Microstructure and geometry "
# Discretization
Ny_cath = 5                         # no. cells in cathode
Ny_sep = 3                          # no. cells in separator

# Geometric/microstructure parameters
th_cath = 60e-6                     # cathode thickness [m]
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

ctifile = 'Li_Air.cti'