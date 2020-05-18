"""

Li-O2 Battery Model:
    This model examines the reactions taking place within the carbon-based
    cathode of a Li-O2 battery. Electrolyte = 1 M LiTFSI in TEGDME

"""

""" Load any needed modules """
"============================================================================"
# Brings in other complex commands as shortcuts
# Shortcut so that code doesn't need to be written out again
import matplotlib.pyplot as plt        # Plotting functions
from scipy.integrate import solve_ivp  # Integrator

""" Read user inputs and initialize variables, vectors, etc. """
"============================================================================"
from li_o2_init import *#caobjs, params, SVptr, pltptr, SV_0, tspan, li_o2_residual
from li_o2_functions import LiO2_func as func

# CLose any pre-exiting open matplotlib files.
plt.close('all')

# Run the model at zero current, to equilibrate:
print('Equilibrating... \n')

# Replace the user-defined current with 0.0 A/m^2
params['i_ext'] = 0.0
SV = solve_ivp(lambda t, y: func(t, y, params, objs, SVptr), [0, t_ocv], \
    SV0, method='BDF', atol=params['atol'], rtol=params['rtol'])

# Save the final state at the initial state for the simulation:
SV0 = SV.y[:,-1]

print('Discharging... \n')
# Re-set the current to the user-provided value and integrate:
params['i_ext'] = i_ext
SV = solve_ivp(lambda t, y: func(t, y, params, objs, SVptr), [0, tspan], \
    SV0, method='BDF', atol=params['atol'], rtol=params['rtol'])

""" Plot solutions to concentrations and potentials """
"============================================================================"

for i in range(Ny_cath):
    SV_move = i * len(SV_single_cath)

    plt.figure(1)
    plt.plot(SV.t,SV.y[SVptr['phi']+SV_move,:],label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('Double Layer Potential (V)')
    plt.tight_layout()
    plt.legend()

    plt.figure(2)
    plt.plot(SV.t,SV.y[SVptr['E_oxide']+SV_move,:],label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('Li2O2 Volume Fraction')
    plt.tight_layout()
    plt.legend()

    plt.figure(3)
    plt.plot(SV.t, SV.y[SVptr['elyte'][pltptr['Li+']]+SV_move,:],label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('Li+ Concentration (kg/m3)')
    plt.tight_layout()
    plt.legend()

    plt.figure(4)
    plt.plot(SV.t, SV.y[SVptr['elyte'][pltptr['O2']]+SV_move,:], label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('O2 Concentration (kg/m3)')
    plt.legend()

    plt.figure(5)
    plt.plot(SV.t, SV.y[SVptr['elyte'][pltptr['PF6-']]+SV_move,:], label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('PF6- Concentration (kg/m3)')
    plt.tight_layout()
    plt.legend()

    plt.figure(6)
    A_carb = params['A_carbon_init'] - SV.y[SVptr['E_oxide']+SV_move,:]*1.5/params['r_oxide']
    plt.plot(SV.t, A_carb, label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('Available surface area per unit volume.')
    plt.tight_layout()
    plt.legend()
    """plt.figure(6)
    plt.plot(SV.t,SV.y[SVptr['elyte'][pltptr['EC']]+SV_move,:],label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('EC Concentration (kg/m3)')
    plt.tight_layout()
    plt.legend()

    plt.figure(7)
    plt.plot(SV.t,SV.y[SVptr['elyte'][pltptr['EMC']]+SV_move,:],label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('EMC Concentration (kg/m3)')
    plt.tight_layout()
    plt.legend()"""

Nplot = np.linspace(1,Ny_sep,Ny_sep)

"""for i in range(len(Nplot)):
    SV_move = i * len(SV_single_sep)

    plt.figure(8)
    plt.plot(SV.t,SV.y[SVptr['sep_phi']+SV_move,:],label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('Double Layer Potential - separator/anode (V)')
    plt.tight_layout()
    plt.legend()

    plt.figure(9)
    plt.plot(SV.t,SV.y[SVptr['sep_elyte'][pltptr['Li+']]+SV_move,:],label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('Li+ Concentration - separator (kg/m3)')
    plt.tight_layout()
    plt.legend()

    plt.figure(10)
    plt.plot(SV.t,SV.y[SVptr['sep_elyte'][pltptr['O2']]+SV_move,:],label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('O2 Concentration - separator (kg/m3)')
    plt.legend()

    plt.figure(11)
    plt.plot(SV.t,SV.y[SVptr['sep_elyte'][pltptr['PF6-']]+SV_move,:],label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('PF6- Concentration - separator (kg/m3)')
    plt.tight_layout()
    plt.legend()

    plt.figure(12)
    plt.plot(SV.t,SV.y[SVptr['sep_elyte'][pltptr['EC']]+SV_move,:],label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('EC Concentration - separator (kg/m3)')
    plt.tight_layout()
    plt.legend()

    plt.figure(13)
    plt.plot(SV.t,SV.y[SVptr['sep_elyte'][pltptr['EMC']]+SV_move,:],label=i+1)
    plt.xlabel('Time (s)')
    plt.ylabel('EMC Concentration - separator (kg/m3)')
    plt.tight_layout()
    plt.legend()"""

plt.show()
