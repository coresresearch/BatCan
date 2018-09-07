"""
Author:
    Amy LeBar (20 August 2018)

Li-O2 Battery Model:
    This model examines the reactions taking place within the carbon-based
    cathode of a Li-O2 battery. Electrolyte = LiPF6?

"""
""" Load any needed modules """
"-------------------------------------------------------------------------"
import os, time
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.misc import derivative

""" BEGIN USER INPUTS """
"-------------------------------------------------------------------------"
phi_elyte = 3               # [V]
Ck_Li2O2 = 0.001            # concentration
Ck_elyte = 0.001            # concentration

#atol = 1e-12                # absolute tolerance for IVP solver (default = 1e-6)
#rtol = 1e-9                 # relative tolerance for IVP solver (default = 1e-3)

tspan = 120                 # [s]

i_ext = 1.15                # A/cm2

Nx = 1                      # 1D model
Ny = 3                      # no. cells in the y-direction
Nvars = 3                # no. of variables per location.
dy = 1/Ny
#i_far = -ct.faraday         # Faradaic current
#i_far = params.z_elyte.getH() * R_elyte * ct.faraday
A_int = 1                   # area of interface
C_dl = 1.1e-2               # double layer capacitance (F/m2)
""" END USER INPUTS  """
"-------------------------------------------------------------------------"
#Initialize the solution vector.
SV_0 = np.tile([phi_elyte, Ck_Li2O2, Ck_elyte],Ny)

params = {'i_ext':i_ext }
# Define function to solve
"Parameter order must be t, SV"
def LiO2_func(t,SV,params):
    # Inital values
    """Note: these will be deleted - the SV_0 should be established outside
        of the funciton, and *passed to* the function, below on line 81"""
    """SV_0 = np.zeros(Ny*Nvars)
    SV_0[0:Ny*Nvars:Nvars] = phi_elyte
    SV_0[1:Ny*Nvars-1:Nvars] = Ck_Li2O2
    SV_0[2:Ny*Nvars-2:Nvars] = Ck_elyte"""
    dSVdt = np.zeros_like(SV)
    dPhidt = np.zeros(Ny)
    dCkLi2O2dt = np.zeros_like(dPhidt)
    dCkElytedt = np.zeros_like(dPhidt)

    i_ext = params['i_ext']     # [A]
    i_io = np.zeros(Ny + 1)     # initialize ionic current vector
    i_el = np.zeros(Ny + 1)     # initialize electronic current vector
    i_el[0] = i_ext             # electronic current at air/cathode boundary
    i_io[-1] = i_ext            # ionic current at cathode/elyte boundary

    i_far = 0                   # Eventually be calculated via Cantera.

    i_dl = np.zeros(Ny)

    for j in range(Ny):
        i_dl[j] = (i_io[j] - i_io[j+1] + i_el[j] - i_el[j+1])/dy - i_far*A_int

    dPhidt = i_dl/(C_dl*A_int)

    for i in range(Ny):
        dSVdt[0 + i*Ny] = dPhidt[i]
        dSVdt[1 + i*Ny] = dCkLi2O2dt[i]
        dSVdt[2 + i*Ny] = dCkElytedt[i]

    return dSVdt


# Solve function using IVP solver

SV = solve_ivp(lambda t, y: LiO2_func(t,y,params), [0, tspan], SV_0)

#print(dSVdt)
print(SV.y)

## Plot solution vector
#for i in range(Ny):
#    plt.plot(SV.t,SV.y[0 + i*Ny,:], label='phi_elyte')
#    plt.plot(SV.t,SV.y[1 + i*Ny,:], label='Ck_Li2O2')
#    plt.plot(SV.t,SV.y[2 + i*Ny,:], label='Ck_elyte')
#    plt.legend(loc = 'best')
#
#print('SV = ')
#print(SV.y)
