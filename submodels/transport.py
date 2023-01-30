"""submodules.transport - Transport-related equations and routines
"""
import cantera as ct
import numpy as np

def dilute_solution(state_1, state_2, sep):
    dy_inv = 1. / ((state_1['dy'] + state_2['dy']))

    C_k_int = ((state_2['dy'] * state_1['C_k'] + state_1['dy'] * state_2['C_k'])
        * dy_inv)
    T_int = ((state_2['dy'] * state_1['T'] + state_1['dy'] * state_2['T'])
        * dy_inv)

    D_scale_1 = sep.scale_diff(state_1['C_k'], sep)
    D_scale_2 = sep.scale_diff(state_2['C_k'], sep)

    D_k_scale_1 = sep.D_k - D_scale_1
    D_k_scale_2 = sep.D_k - D_scale_2

    D_k_eff = (state_2['dy']*D_k_scale_1*state_1['microstructure'] +
                state_1['dy']*D_k_scale_2*state_2['microstructure'])*dy_inv

    D_k_mig = (D_k_eff * sep.elyte_obj.charges * ct.faraday * C_k_int
        / ct.gas_constant / T_int)

    # Dilute solution theory fluxes:
    N_k_elyte = ((D_k_eff * (state_1['C_k'] - state_2['C_k'])
        + D_k_mig * (state_1['phi'] - state_2['phi'])) * 2 * dy_inv)

    # Ionic current = sum(z_k*N_k*F)
    i_io = ct.faraday*np.dot(N_k_elyte, sep.elyte_obj.charges)

    return N_k_elyte, i_io

def scale_diff_zhang(C_k, sep):
    # Concentration dependent diffusion scaling based on the model used in
    #   Zhang T., Marinescu M., O'Neill L., Wild M. and Offer G. 2015
    #   Phys. Chem. Chem. Phys. 17 22581
    D_vec = np.zeros_like(C_k)
    #TODO #63 
    C_Li = C_k[sep.index_Li] + sep.flag_lithiated*2*np.sum(C_k[4:])
    D_scale = sep.D_scale_coeff*abs(sep.C_Li_0 - C_Li)
    D_vec[sep.index_Li] = D_scale

    return D_vec

def scale_diff_ideal(C_k, sep):
    # For the ideal case, 
    D_vec = np.zeros_like(sep.D_k)
    return D_vec

def radial_flux(X_Li, sdot_Li, ed):
    # Radial flux of intercalated Li.
    #   X_Li: array of intercalated Li mole fractions (1 per radial volume)
    #   sdot_Li: production of intercalated Li at particle surface (kmol/m2/s)
    #   ed: electrode object
    N_r_Li = np.zeros([ed.nshells+1])
    r_int = 0.5 * (ed.r_shell[:-1] + ed.r_shell[1:])
    N_r_Li[1:-1] = ((ed.r_shell[:-1]*X_Li[:-1] - ed.r_shell[1:]*X_Li[1:])
        * ed.D_Li_ed / ed.dr / r_int)

    # Flux at the surface (in positive r direction) is equal and opposite to 
    #   the production rate from the intercalation reaction:
    N_r_Li[-1] = -sdot_Li
    
    return N_r_Li
