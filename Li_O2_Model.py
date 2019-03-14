"""
Author:
Amy LeBar (20 August 2018)

Li-O2 Battery Model:
This model examines the reactions taking place within the carbon-based
cathode of a Li-O2 battery. Electrolyte = 1 M LiTFSI in TEGDME

"""
""" Load any needed modules """
"============================================================================"
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

""" BEGIN USER INPUTS """
"============================================================================"
phi_elyte_init = -3.19                  # double layer voltage [V]
theta_part_init = 0.                    # initial surface coverage [-]
E_elyte_init = 0.5                      # initial electrolyte volume fraction [-]
E_oxide_init = 1e-12                    # initial oxide volume fraction [-]
E_binder_init = 0.                      # initial binder volume fraction [-]
E_carbon = 1. - E_elyte_init - E_binder_init - E_oxide_init      # initial carbon volume fraction [-]

atol = 1e-10
rtol = 1e-7

tspan = 5306 #7824                                  # [s]

i_ext = -1                                          # [A/m2]

Nx = 1                                              # 1D model
Ny = 1                                              # no. cells in the y-direction
Nvars = 4                                           # no. of variables
th_ca = 110e-6                                      # cathode thickness [m]
dy = th_ca/Ny                                       # [m]
d_part = 10e-6                                      # carbon particle diameter [m]
d_oxide = 2e-6                                      # oxide particle diameter [m]
th_oxide = 0.5e-6                                   # thickness of oxide ellipsoid [m]
V_part = 4/3 * np.pi * (d_part / 2)**3              # particle volume [m3]
A_part = 4 * np.pi * (d_part / 2)**2                # particle surface area [m2]
A_int = E_carbon * A_part / V_part                  # interface area [m2/m3 total]
A_oxide = np.pi * d_oxide**2 / 4                    # oxide area contacting carbon particle
V_oxide = 2/3 * np.pi * (d_oxide/2)**2 * th_oxide   # oxide volume [m3]
A_grid = 6.65e-18                                   # area per grid [m2]
A_int_grid = E_carbon * A_grid /V_part              # grid area per volume [m2/m3]
C_dl = 1.1e-6                                       # double layer capacitance [F/m2]
sigma_ca = 75.0                                     # bulk cathode electrical conductivity [S/m]
elyte_charge = np.array([0, 1, -1, 0, 0])           # elyte species charge

TP = 300, 101325                                    # inital temp, pressure [K, Pa]

ctifile = 'LiAir.cti'

""" END USER INPUTS """
"============================================================================"
# Import necessary phases
gas = ct.Solution(ctifile,'air')
cath_b = ct.Solution(ctifile,'graphite')
elyte = ct.Solution(ctifile,'electrolyte')
#oxide = ct.Solution(ctifile,'Li2O2')
inter = ct.Interface(ctifile,'cathode_surf',[elyte,cath_b])
air_elyte = ct.Interface(ctifile,'air_elyte',[gas,elyte])
Li_b = ct.Solution(ctifile,'Lithium')
Li_s = ct.Interface(ctifile,'Li_surface',[Li_b,elyte])

oxide.TP = TP
elyte.TP = TP
inter.TP = TP
cath_b.TP = TP

# Store these phases in a common 'objs' dict
objs = {}
objs['gas'] = gas
objs['cath_b'] = cath_b
objs['elyte'] = elyte
objs['oxide'] = oxide
objs['inter'] = inter
objs['air_elyte'] = air_elyte
objs['Li_b'] = Li_b
objs['Li_s'] = Li_s

# Store parameters in a common 'params' dict
params = {}
params['i_ext'] = i_ext
params['T'] = TP[0]
params['E_elyte_0'] = E_elyte_init
params['E_oxide_0'] = E_oxide_init
params['rtol'] = rtol
params['atol'] = atol
params['vaca'] = 4.73e7

# Store pointers in a common 'ptr' dict
ptr = {}
ptr['elec'] = elyte.n_species + oxide.n_species         # electron in the inter net_production_rates vector
ptr['oxide'] = elyte.n_species                          # oxide in the inter net_production_rates vector
ptr['elyte'] = np.arange(0,elyte.n_species)             # electrolyte in the inter net_production_rates vector

# Store solution vector pointers in a common 'SVptr' dict
SVptr = {}
SVptr['phi'] = 0                                        # double layer potential in solution vector SV
SVptr['oxide'] = 1                                      # oxide density in solution vector SV
SVptr['elyte'] = np.arange(2,elyte.n_species + 2)       # electrolyte densities in solution vector SV
SVptr['theta'] = 2 + len(SVptr['elyte'])                # surface coverage in SV

# Store plot pointers in a common 'pltptr' dict
pltptr = {}
pltptr['O2'] = 2
pltptr['Li+'] = 3
pltptr['PF6-'] = 4
pltptr['EC'] = 5
pltptr['EMC'] = 6

# Set inital values
rho_oxide_init = oxide.density*params['E_oxide_0']                          # oxide concentraion
rho_elyte_init = elyte.Y*elyte.density*params['E_elyte_0']                  # electrolyte concentrations
SV0 = np.r_[phi_elyte_init,rho_oxide_init,rho_elyte_init,theta_part_init]   # store in an array
SV_0 = np.tile(SV0,Ny)                                                      # tile SV0 based on discritization

# Define function to solve
def LiO2_func(t,SV,params,objs,ptr,SVptr):

    dSVdt = np.zeros_like(SV)
    dPhidt = np.zeros_like(SV)
    dRhoOxidedt = np.zeros_like(SV)
    dRhoElytedt = np.zeros_like(SV)
    dThetadt = np.zeros_like(SV)

    # Pull phases out of 'objs' inside function
    gas = objs['gas']
    cath_b = objs['cath_b']
    elyte = objs['elyte']
    oxide = objs['oxide']
    inter = objs['inter']
    air_elyte = objs['air_elyte']
    Li_b = objs['Li_b']
    Li_s = objs['Li_s']

    # Set electronic and ionic currents, and flux terms
    i_ext = params['i_ext']     # [A]
    i_io = np.zeros(Ny + 1)     # initialize ionic current vector
    i_el = np.zeros(Ny + 1)     # initialize electronic current vector
    i_el[0] = i_ext             # electric current at air/cathode boundary
    i_io[-1] = i_ext            # ionic current at cathode/elyte

    J_in = np.zeros(elyte.n_species)
    J_out = np.zeros(elyte.n_species)

    # Set potentials
    cath_b.electric_potential = 0
    oxide.electric_potential = 0
    elyte.electric_potential = SV[SVptr['phi']]

    # Set mass fractions and electrolyte properties
    E_oxide = SV[SVptr['oxide']] / oxide.density_mass               # oxide volume fraction
    E_elyte = params['E_elyte_0'] - (E_oxide - params['E_oxide_0'])
    rho_elyte = (sum(SV[SVptr['elyte']])) / E_elyte
    elyte.TDY = params['T'], rho_elyte, SV[SVptr['elyte']]

    J_in = air_elyte.get_net_production_rates(elyte)
    W_elyte = elyte.molecular_weights

# ---------- Different methods for calculating J_out ----------
#    J_out[1] = i_io[-1] * W_elyte[1] / ct.faraday
#    J_out = W_elyte * ct.faraday * i_io[-1]
#    J_out = ct.faraday * elyte_charge * i_ext
#    J_out = i_ext / ct.faraday * elyte.molecular_weights
#    J_out = ct.faraday * elyte.molecular_weights / elyte_charge
#    J_out = elyte.molecular_weights / elyte_charge
#    J_out = i_io[-1] / ct.faraday * elyte.molecular_weights / elyte_charge
#    print('J_in =',J_in)
#    print('J_out =',J_out)

    # Calculate net production rates at interface
    sdot = inter.net_production_rates                 # interface production rates

    # Calculate Faradaic current
    i_far = -sdot[ptr['elec']] * ct.faraday           # Faradaic current

    # Calculate change in oxide concentration
    W_oxide = oxide.mean_molecular_weight             # oxide molecular weight
    #A_int_avail = A_int - E_oxide / th_oxide          # available interface area on carbon particle
    A_int_avail = A_int * (1 - SVptr['theta'])        # available interface area on carbon particle (new method)

    dRhoOxidedt = sdot[ptr['oxide']] * A_int_avail * W_oxide

    # Calculate change in double layer potential
    i_dl = (i_io[0] - i_io[-1]) / dy - i_far*A_int_avail   # double layer current
    dPhidt = i_dl / (C_dl*A_int_avail)                           # double layer potential

    # Calculate change in electrolyte concentrations
    dRhoElytedt = (J_in - J_out) / dy + (sdot[ptr['elyte']] * A_int_avail * W_elyte)

    # Calculate change in vacancies on carbon particles
    E_part = 1 - E_oxide - E_elyte                          # volume fraction of carbon particles
    vaca_0 = params['vaca'] / V_part * E_part               # initial number of vacancies per m3
    dThetadt = (-sdot[ptr['oxide']]/20 * ct.avogadro * A_int_grid) / vaca_0

    # Load differentials into dSVdt
    dSVdt[SVptr['phi']] = dPhidt                            # double layer potential
    dSVdt[SVptr['oxide']] = dRhoOxidedt                     # oxide concentration
    dSVdt[SVptr['elyte']] = dRhoElytedt                     # electrolyte concentration
    dSVdt[SVptr['theta']] = dThetadt                        # particle surface coverages

    return dSVdt

print('Running model')

# Solve function using IVP solver
SV = solve_ivp(lambda t, y: LiO2_func(t,y,params,objs,ptr,SVptr), [0, tspan], SV_0, method='BDF',atol=params['atol'],rtol=params['rtol'])

""" Plot solutions to concentrations and potentials """
"============================================================================"
plt.figure(1)
plt.plot(SV.t,SV.y[SVptr['phi']])
plt.xlabel('Time (s)')
plt.ylabel('Double Layer Potential (V)')

plt.figure(2)
plt.plot(SV.t,SV.y[SVptr['oxide']])
plt.xlabel('Time (s)')
plt.ylabel('Oxide Concentration (kg/m3)')

E_oxide = SV.y[SVptr['oxide']] / oxide.density_mass      # oxide volume fraction
E_elyte = params['E_elyte_0'] - (E_oxide - params['E_oxide_0'])

plt.figure(3)
plt.plot(SV.t,E_elyte)
plt.xlabel('Time (s)')
plt.ylabel('Elyte Volume Fraction')

A_int_avail = A_int - A_int * (1 - SV.y[SVptr['theta']])

plt.figure(4)
plt.plot(SV.t,A_int_avail)
plt.xlabel('Time (s)')
plt.ylabel('Available Area (m2)')

plt.figure(5)
plt.plot(SV.t,SV.y[pltptr['Li+']])
plt.xlabel('Time (s)')
plt.ylabel('Li+ Concentration (kg/m3)')

plt.figure(6)
plt.plot(SV.t,SV.y[pltptr['O2']])
plt.xlabel('Time (s)')
plt.ylabel('O2 Concentration (kg/m3)')

plt.figure(7)
plt.plot(SV.t,SV.y[-1])
plt.xlabel('Time (s)')
plt.ylabel('Surface Coverage')

plt.show()