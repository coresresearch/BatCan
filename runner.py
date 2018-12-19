# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:26:02 2018

@author: Amy LeBar
"""
import numpy as np
import pylab as plt

from Li2O2_func_concentrations import runner as func

""" -------------------------------- C/D -------------------------------- """
i_ext = [-1.08674e-3,-2.173e-3,-1.0865e-3,-0.019561,-0.03912]
tspan = [7200,3600,720,400,200]
C = [0.5,1,5,9,18]

V = np.zeros_like(i_ext)

for index,current in enumerate(i_ext):
    V[index] = func(current,tspan[index])
    plt.figure()
    plt.plot(func.t/3600 * current,V[index])


" ----------------------------------------------------------------------- "

""" ------------------------- Particle Diameter -------------------------"""
#d_part = np.linspace(1e-3,1e-9,3)
#i_ext = np.linspace(0,-1e-3,500)           # external current [A/m2]
#

#
#z = (len(d_part),len(i_ext))
#
#Phi_dl = np.zeros(z)
#
#for index,diam in enumerate(d_part):
#    for dex,current in enumerate(i_ext):
#        Phi_dl[index,dex] = func(current,diam)
# 
#    plt.plot(i_ext,Phi_dl[index,:])
#    plt.xlim(1e-6, -1.01e-3)
#    plt.ylim(np.max(Phi_dl)-0.001,np.min(Phi_dl)+0.001)
#    plt.xlabel('Current Density (A/m2)')
#    plt.ylabel('Voltage (V)')
#
#plt.legend(['1e-3 m','1e-6 m','1e-9 m'],loc='lower left')
#plt.show()
" ----------------------------------------------------------------------- "

## Plot solutions to concentrations and potentials
#plt.semilogx(SV.t,SV.y[SVptr['phi']])
#plt.xlabel('Time (s)')
#plt.ylabel('Double Layer Potential (V)')
#plt.show()
#
#plt.plot(SV.t,SV.y[SVptr['oxide']],'m')
#plt.xlabel('Time (s)')
#plt.ylabel('Oxide Concentration (kg/m3)')
#plt.show()