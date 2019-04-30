"""
Author:
Amy LeBar (23 April 2019)

Runner file for function Li_O2_Model.

Input: initial values, tolerance parameters, discretization, geometry, properities

"""
""" Load any needed modules """
"============================================================================"
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from Li_O2_Model import LiO2_func as func


""" BEGIN USER INPUTS """
"============================================================================"
# Initial values
phi_elyte_init = -3.19                  # double layer voltage [V]
theta_init = 0.                         # initial surface coverage [-]
E_elyte_init = 0.5                      # initial electrolyte volume fraction [-]
E_oxide_init = 1e-12                    # initial oxide volume fraction [-]
E_binder_init = 0.                      # initial binder volume fraction [-]
E_carbon = 1. - E_elyte_init - E_binder_init - E_oxide_init      # initial carbon volume fraction [-]

# Tolerancing
atol = 1e-8
rtol = 1e-6

# Simulation time and external current
tspan = 1e4#3.15e5                                 # [s]
i_ext = -1e-4                                       # [A/m2]

# Model parameters
Nx = 1                                              # 1D model
Ny = 3                                              # no. cells in the y-direction
Nvars = 3                                           # no. of variables

# Geometric parameters
th_ca = 110e-6                                      # cathode thickness [m]
dyInv = Ny/th_ca                                    # inverse dy [1/m]
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

# Properties
E_carb = 0.8                                        # carbon paper porosity
E_carb_inv = 1/E_carb                               # inverse of carbon porosity
C_dl = 1.1e-6                                       # double layer capacitance [F/m2]
sigma_ca = 75.0                                     # bulk cathode electrical conductivity [S/m]
Zk_elyte = np.array([0, 1, -1, 0, 0])               # elyte species charge
sigma_Li = -0.05e2                                  # ionic conductivity Li+ [S/m]
sigma_PF6 = 10.8*0.1                                # ionic conductivity PF6- salt [S/m]

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

# Set phase temperatures
elyte.TP = TP
inter.TP = TP
cath_b.TP = TP

# Define electrolyte ionic conductivities
sigma_io = np.zeros_like(elyte.X)
sigma_io[1] = sigma_Li
sigma_io[2] = sigma_PF6

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
params['sigma'] = sigma_io
params['rtol'] = rtol
params['atol'] = atol
params['Ny'] = Ny
params['dyInv'] = dyInv
params['A_int'] = A_int
params['C_dl'] = C_dl
params['E_carb_inv'] = E_carb_inv
params['Zk_elyte'] = Zk_elyte

# Store pointers in a common 'ptr' dict
ptr = {}
ptr['elec'] = elyte.n_species                           # electron in the inter net_production_rates vector
ptr['elyte'] = np.arange(0,elyte.n_species)             # electrolyte in the inter net_production_rates vector
ptr['O2_elyte'] = 0                                     # location of O2 in elyte phase
ptr['Li+'] = 1                                          # location of Li+ in elyte phase
ptr['PF6-'] = 2                                         # location of PF6- in elyte phase

# Store solution vector pointers in a common 'SVptr' dict
SVptr = {}
SVptr['phi'] = 0                                        # double layer potential in solution vector SV
SVptr['elyte'] = np.arange(1, elyte.n_species + 1)      # electrolyte densities in solution vector SV
SVptr['theta'] = np.arange(6, len(inter.X) + 6)         # surface coverage in SV

# Store plot pointers in a common 'pltptr' dict
pltptr = {}
pltptr['O2'] = 1
pltptr['Li+'] = 2
pltptr['PF6-'] = 3
pltptr['EC'] = 4
pltptr['EMC'] = 5

# Set inital values
rho_elyte_init = elyte.Y*elyte.density                            # electrolyte concentrations
theta_init = [0, 1]                                             # surface coverages
SV_single = np.r_[phi_elyte_init,rho_elyte_init,theta_init]     # store in an array
SV_0 = np.tile(SV_single,Ny)
params['SV_single'] = len(SV_single)                            # put length of single SV into 'params' for indexing

# Solve function using IVP solver
SV = solve_ivp(lambda t, y: func(t,y,params,objs,ptr,SVptr), [0, tspan], SV_0, method='BDF',atol=params['atol'],rtol=params['rtol'])

""" Plot solutions to concentrations and potentials """
"============================================================================"
SV_move = 0*len(SV_single)

plt.figure(1)
plt.plot(SV.t,SV.y[SVptr['phi']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('Double Layer Potential (V)')

plt.figure(2)
plt.plot(SV.t,SV.y[SVptr['theta'][0]+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('Li2O2 Surface Coverage')

plt.figure(3)
plt.plot(SV.t,SV.y[pltptr['Li+']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('Li+ Concentration (kg/m3)')

plt.figure(4)
plt.plot(SV.t,SV.y[pltptr['O2']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('O2 Concentration (kg/m3)')

plt.figure(5)
plt.plot(SV.t,SV.y[pltptr['PF6-']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('PF6- Concentration (kg/m3)')

plt.figure(6)
plt.plot(SV.t,SV.y[pltptr['EC']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('EC Concentration (kg/m3)')

plt.figure(7)
plt.plot(SV.t,SV.y[pltptr['EMC']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('EMC Concentration (kg/m3)')

SV_move = 1*len(SV_single)

plt.figure(1)
plt.plot(SV.t,SV.y[SVptr['phi']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('Double Layer Potential (V)')

plt.figure(2)
plt.plot(SV.t,SV.y[SVptr['theta'][0]+SV_move],'o')
plt.xlabel('Time (s)')
plt.ylabel('Li2O2 Surface Coverage')

plt.figure(3)
plt.plot(SV.t,SV.y[pltptr['Li+']+SV_move],'o')
plt.xlabel('Time (s)')
plt.ylabel('Li+ Concentration (kg/m3)')

plt.figure(4)
plt.plot(SV.t,SV.y[pltptr['O2']+SV_move],'o')
plt.xlabel('Time (s)')
plt.ylabel('O2 Concentration (kg/m3)')

plt.figure(5)
plt.plot(SV.t,SV.y[pltptr['PF6-']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('PF6- Concentration (kg/m3)')

plt.figure(6)
plt.plot(SV.t,SV.y[pltptr['EC']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('EC Concentration (kg/m3)')

plt.figure(7)
plt.plot(SV.t,SV.y[pltptr['EMC']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('EMC Concentration (kg/m3)')
plt.legend(['1st','2nd','3rd'])

SV_move = 2*len(SV_single)

plt.figure(1)
plt.plot(SV.t,SV.y[SVptr['phi']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('Double Layer Potential (V)')
plt.legend(['air/elyte','internal','separator'])

plt.figure(2)
plt.plot(SV.t,SV.y[SVptr['theta'][0]+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('Li2O2 Surface Coverage')
plt.legend(['air/elyte','internal','separator'])

plt.figure(3)
plt.plot(SV.t,SV.y[pltptr['Li+']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('Li+ Concentration (kg/m3)')
plt.legend(['air/elyte','internal','separator'])

plt.figure(4)
plt.plot(SV.t,SV.y[pltptr['O2']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('O2 Concentration (kg/m3)')
plt.legend(['air/elyte','internal','separator'])

plt.figure(5)
plt.plot(SV.t,SV.y[pltptr['PF6-']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('PF6- Concentration (kg/m3)')
plt.legend(['air/elyte','internal','separator'])

plt.figure(6)
plt.plot(SV.t,SV.y[pltptr['EC']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('EC Concentration (kg/m3)')
plt.legend(['air/elyte','internal','separator'])

plt.figure(7)
plt.plot(SV.t,SV.y[pltptr['EMC']+SV_move])
plt.xlabel('Time (s)')
plt.ylabel('EMC Concentration (kg/m3)')
plt.legend(['air/elyte','internal','separator'])

plt.show()