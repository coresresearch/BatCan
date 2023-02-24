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
    C_Li = np.dot(sep.n_Li_atoms, C_k)
    D_scale = sep.D_scale_coeff*abs(sep.C_Li_0 - C_Li)
    D_vec[sep.index_Li] = D_scale

    return D_vec

def scale_diff_ideal(C_k, sep):
    # For the ideal case,
    D_vec = np.zeros_like(sep.D_k)
    return D_vec

def radial_flux(C_k, sdot_k, ed):
    # Radial flux of intercalated Li.
    #   C_k: matrix of electrode species concentrations [kmol/m3]. One array
    #       per radial volume.
    #   sdot_k: species production rates at particle surface (kmol/m2/s)
    #   ed: electrode object

    # Initialize return vector (flux at particle center = 0)
    N_r_Li = np.zeros((ed.n_r + 1, ed.bulk_obj.n_species))
    N_r_Li[1:-1,:] = (C_k[:-1,:] - C_k[1:,:]) * ed.D_k / ed.dr[:-1]

    # Flux at the surface (in positive r direction) is equal and opposite to
    #   the production rate from surface reactions:
    N_r_Li[-1,:] = -sdot_k

    return N_r_Li
