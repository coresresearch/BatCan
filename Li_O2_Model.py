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
theta_init = 0.                         # initial surface coverage [-]
E_elyte_init = 0.5                      # initial electrolyte volume fraction [-]
E_oxide_init = 1e-12                    # initial oxide volume fraction [-]
E_binder_init = 0.                      # initial binder volume fraction [-]
E_carbon = 1. - E_elyte_init - E_binder_init - E_oxide_init      # initial carbon volume fraction [-]

atol = 1e-8
rtol = 1e-6

tspan = 3.15e5                                      # [s]

i_ext = -1e-4#-1                                    # [A/m2]

Nx = 1                                              # 1D model
Ny = 1                                              # no. cells in the y-direction
Nvars = 3                                           # no. of variables
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

ctifile = 'Li_Air.cti'

""" END USER INPUTS """
"============================================================================"
# Import necessary phases
gas = ct.Solution(ctifile,'air')
cath_b = ct.Solution(ctifile,'graphite')
elyte = ct.Solution(ctifile,'electrolyte')
oxide = ct.Solution(ctifile,'Li2O2')
inter = ct.Interface(ctifile,'cathode_surf',[elyte,cath_b])
air_elyte = ct.Interface(ctifile,'air_elyte',[gas,elyte])
Li_b = ct.Solution(ctifile,'Lithium')
Li_s = ct.Interface(ctifile,'Li_surface',[Li_b,elyte])

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

# Store pointers in a common 'ptr' dict
ptr = {}
ptr['elec'] = elyte.n_species                           # electron in the inter net_production_rates vector
ptr['elyte'] = np.arange(0,elyte.n_species)             # electrolyte in the inter net_production_rates vector

# Store solution vector pointers in a common 'SVptr' dict
SVptr = {}
SVptr['phi'] = 0                                                    # double layer potential in solution vector SV
SVptr['elyte'] = np.arange(1, elyte.n_species + 1)                  # electrolyte densities in solution vector SV
SVptr['theta'] = np.arange(6, len(inter.X) + 6)  # surface coverage in SV

# Store plot pointers in a common 'pltptr' dict
pltptr = {}
pltptr['O2'] = 1
pltptr['Li+'] = 2
pltptr['PF6-'] = 3
pltptr['EC'] = 4
pltptr['EMC'] = 5

# Set inital values
#rho_oxide_init = oxide.density*params['E_oxide_0']              # oxide concentraion
rho_elyte_init = elyte.Y*elyte.density*params['E_elyte_0']      # electrolyte concentrations
theta_init = [0, 1]                                             # surface coverages
SV0 = np.r_[phi_elyte_init,rho_elyte_init,theta_init]           # store in an array
SV_0 = np.tile(SV0,Ny)                                          # tile SV0 based on discritization

# Define function to solve
def LiO2_func(t,SV,params,objs,ptr,SVptr):
#    print('t =',t)

    dSVdt = np.zeros_like(SV)
    dPhidt = np.zeros_like(SV)
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

    J_in = np.zeros(elyte.n_species)        # initialize species 'in' vector
    J_out = np.zeros(elyte.n_species)       # initialize species 'out' vector
    
    W_elyte = elyte.molecular_weights

    J_out[1] = i_io[-1] / ct.faraday / 1 * W_elyte[1]
    

    # Set potentials and surface coverages
    cath_b.electric_potential = 0
#    oxide.electric_potential = 0
    elyte.electric_potential = SV[SVptr['phi']]
    elyte.Y = SV[SVptr['elyte']]
    inter.coverages = SV[SVptr['theta']]

    # Calculate Faradaic current
    i_far = inter.get_net_production_rates(cath_b) * ct.faraday    # Faradaic current
    
    # Calculate available area
    A_int_avail = A_int #* inter.X[0]                           # available interface area on carbon particle (new method)
    
    # Calculate change in double layer potential
    i_dl = (i_io[0] - i_io[-1]) / dy + i_far*A_int_avail        # double layer current
    dPhidt = i_dl / (C_dl*A_int)                                # double layer potential

    # Calculate change in electrolyte concentrations
    J_in = air_elyte.get_net_production_rates(elyte)
    sdot_int = inter.get_net_production_rates(elyte) * A_int_avail
    sdot_dl = np.zeros_like(sdot_int)
    sdot_dl[1] = -i_dl / (ct.faraday*1)
    dRhoElytedt = (J_in - J_out) / dy + (sdot_int + sdot_dl) * W_elyte
    
    # Calculate change in surface coverages
    dThetadt = inter.get_net_production_rates(inter) / inter.site_density

    # Load differentials into dSVdt
    dSVdt[SVptr['phi']] = dPhidt                            # double layer potential
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
plt.plot(SV.t,SV.y[SVptr['theta'][0]])
plt.xlabel('Time (s)')
plt.ylabel('Li2O2 Surface Coverage')

plt.figure(3)
plt.plot(SV.t,SV.y[pltptr['Li+']])
plt.xlabel('Time (s)')
plt.ylabel('Li+ Concentration (kg/m3)')

plt.figure(4)
plt.plot(SV.t,SV.y[pltptr['O2']])
plt.xlabel('Time (s)')
plt.ylabel('O2 Concentration (kg/m3)')

plt.figure(5)
plt.plot(SV.t,SV.y[pltptr['PF6-']])
plt.xlabel('Time (s)')
plt.ylabel('PF6- Concentration (kg/m3)')

plt.figure(6)
plt.plot(SV.t,SV.y[pltptr['EC']])
plt.xlabel('Time (s)')
plt.ylabel('EC Concentration (kg/m3)')

plt.figure(7)
plt.plot(SV.t,SV.y[pltptr['EMC']])
plt.xlabel('Time (s)')
plt.ylabel('EMC Concentration (kg/m3)')

plt.show()