"""submodules.transport - Transport-related equations and routines
"""
import cantera as ct
import numpy as np

def dilute_solution(state_1, state_2, sep):
    dy_inv = 1. / (state_1['dy'] + state_2['dy'])

    C_k_int = ((state_1['dy'] * state_1['C_k'] + state_2['dy'] * state_2['C_k'])
        * dy_inv)
    T_int = ((state_1['dy'] * state_1['T'] + state_2['dy'] * state_2['T'])
        * dy_inv)

    D_k_eff = ((state_1['dy'] * state_1['microstructure'] 
        + state_2['dy'] * state_2['microstructure']) * sep.D_k * dy_inv)
    D_k_mig = (D_k_eff * sep.elyte_obj.charges * ct.faraday * C_k_int
        / ct.gas_constant / T_int)
    
    # Dilute solution theory fluxes:
    N_k_elyte = ((D_k_eff * (state_1['C_k'] - state_2['C_k'])
        + D_k_mig * (state_1['phi'] - state_2['phi'])) * 2 * dy_inv)
    
    # Ionic current = sum(z_k*N_k*F)
    i_io = ct.faraday*np.dot(N_k_elyte, sep.elyte_obj.charges)

    return N_k_elyte, i_io