""" Functions for calculating Li-O2 battery performance """

" ================== Load needed modules =================="
import numpy as np      # Support for multidimensional arrays and functions
import cantera as ct    # Open source chemistry toolbox

import sys
sys.path.append('../')

def LiO2_func(t, SV, params, objs, SVptr):

    # Import diffusion coefficient calculator:
    if params['transport'] == 'cst':
        from functions.diffusion_coeffs import cst as diff_coeffs
    elif params['transport'] == 'dst':
        from functions.diffusion_coeffs import dst as diff_coeffs
    else:
        raise Exception('Please specify a valid transport model: cst or dst')

    # Initialize the residual vector:
    dSVdt = np.zeros_like(SV)

    #  Pull phase objects out of storage in 'objs', for convenient use:
    gas, ca_bulk, elyte, oxide, ca_surf, tpb, air_elyte, Li_bulk, Li_surf = \
        read_cantera_objs(objs)
    
    # Read out and store some constants locally:
    # Faraday's constant (C/kmol e-)
    F = ct.faraday
    # oxide molar volume
    Vbar_oxide = oxide.mean_molecular_weight / oxide.density_mass

    # Pull parameters out of 'params' inside function
    i_ext = params['i_ext']

    " ============================ CATHODE ============================ "
    
    # Initialize electrolyte species production rates due to double layer 
    #    current:
    sdot_dl = np.zeros_like(elyte.X)

    # Read out conditions for first node (adjacent to current collector)
    j = 0
    phi_elyte, eps_oxide, C_k_elyte = read_state(SV, SVptr, j)
    # Set the state of the relevant Cantera objects:
    ca_bulk, oxide, elyte, gas = set_states(phi_elyte, C_k_elyte, \
        params, ca_bulk, oxide, elyte, gas)
    # Calculate electrolyte volume fraction
    eps_elyte = 1. - params['eps_carbon'] - eps_oxide

    # Array of molar fluxes (kmol/m2/s) and ionic current (A/m2) at air/elyte BC
    N_k_in = air_elyte.get_net_production_rates(elyte)
    i_io_in = 0.

    # Initialize array of molar fluxes (kmol/m2/s) out of the volume:
    N_k_out = np.zeros_like(N_k_in)

    for j in np.arange(params['N_y_ca'] - 1):

        # Area of carbon surface per unit volume:
        A_carb = params['A_carbon_init'] - eps_oxide*1.5/params['r_oxide']

        # Calculate chemical & electron production terms (kmol or A per m^2-s):
        i_far = tpb.get_net_production_rates(ca_bulk) * F * A_carb
        sdot_elyte_surf = tpb.get_net_production_rates(elyte) * A_carb
        i_far = sum(sdot_elyte_surf*params['Z_k_elyte']) * F
        sdot_oxide = tpb.get_net_production_rates(oxide) * A_carb

        # Elyte state for 'next node'
        phi_elyte_next, eps_oxide_next, C_k_elyte_next = \
            read_state(SV, SVptr, j+1)
            
        # Set the state of the relevant Cantera objects:
        ca_bulk, oxide, elyte, gas = set_states(phi_elyte_next, \
            C_k_elyte_next, params, ca_bulk, oxide, elyte, gas)
        # Calculate electrolyte volume fraction
        eps_elyte_next = 1. - params['eps_carbon'] - eps_oxide_next

        # Concentration and volume fracion at interface between nodes:
        C_k_elyte_int = 0.5*(C_k_elyte + C_k_elyte_next)
        eps_elyte_int = 0.5*(eps_elyte + eps_elyte_next)

        # Read out diffusion coefficients.  First array returned represents the 
        #    chemical diffusion coefficients.  Second array returned represents 
        #    the migration diffusion coefficients.
        D_k_elyte = diff_coeffs(C_k_elyte_int, objs, params)

        N_k_out = (D_k_elyte[0] * (C_k_elyte - C_k_elyte_next) + \
            D_k_elyte[1]*(phi_elyte - phi_elyte_next))*params['dyInv_ca'] \
            *eps_elyte_int**params['bruggman']

        # Ionic current out of node
        i_io_out = F * sum(params['Z_k_elyte'] * N_k_out)

        # Calculate change in double layer potential
        #    Double layer current
        i_dl = (i_io_in - i_io_out)*params['dyInv_ca'] + i_far
        dSVdt[SVptr['phi_dl'][j]] = i_dl / (params['C_dl']*A_carb)

        # Calculate change in electrolyte concentrations
        #   Double layer current acts as a source/sink for one species:
        sdot_dl[params['i_dl_species']] = -i_dl / \
            (F * params['Z_k_elyte'][params['i_dl_species']])
        
        dSVdt[SVptr['C_k_elyte_ca'][j]] = ((N_k_in - N_k_out) \
            * params['dyInv_ca'] \
            + (sdot_elyte_surf + sdot_dl)) / eps_elyte

        # Calculate change in oxide volume fraction
        dSVdt[SVptr['eps_oxide'][j]] = Vbar_oxide*sdot_oxide

        # Set next 'in' fluxes and ionic current equal to current 'out' fluxes:
        N_k_in = N_k_out
        i_io_in = i_io_out

        # Set electrolyte properties equal to current "next" node properties:
        phi_elyte = phi_elyte_next
        C_k_elyte = C_k_elyte_next
        eps_elyte = eps_elyte_next

    " --- Volume adjacent to the separator --- "
    # Set previous 'next' as current 'this'
    j = params['N_y_ca'] - 1

    # Area of carbon surface per unit volume:
    A_carb = params['A_carbon_init'] - eps_oxide*1.5/params['r_oxide']

    # Calculate chemical & electron production terms (kmol or A per m^2-s):
    i_far = tpb.get_net_production_rates(ca_bulk) * A_carb * F
    sdot_elyte_surf = tpb.get_net_production_rates(elyte) * A_carb
    sdot_oxide = tpb.get_net_production_rates(oxide) * A_carb

    # Charge neutrality on the cathode as a whole:
    i_io_out = i_ext

    # TODO: #8 replace this with electro-diffusive flux:
    N_k_out = np.zeros_like(N_k_in)
    N_k_out[params['i_dl_species']] = i_io_out / \
        (F * params['Z_k_elyte'][params['i_dl_species']])

    # Calculate change in double layer potential
    #    Double layer current
    i_dl = (i_io_in - i_io_out)*params['dyInv_ca'] + i_far
    dSVdt[SVptr['phi_dl'][j]] = i_dl / (params['C_dl']*A_carb)

    # Double layer current acts as a source/sink for one species:
    sdot_dl[params['i_dl_species']] = -i_dl\
        / (F*params['Z_k_elyte'][params['i_dl_species']])
    
    # Calculate change in electrolyte concentrations
    dSVdt[SVptr['C_k_elyte_ca'][j]] = ((N_k_in - N_k_out)*params['dyInv_ca']\
        + (sdot_elyte_surf + sdot_dl)) / eps_elyte

    # Change in oxide volume fraction:
    dSVdt[SVptr['eps_oxide'][j]] = Vbar_oxide*sdot_oxide


    " ============================== SEPARATOR ========================= "
    # Initialize SV pointer offset for separator


    """

    " --- Pre-loop --- "
    elyte.TDY = T, abs(sum(SV[SVptr['sep_elyte']])), SV[SVptr['sep_elyte']]/abs(sum(SV[SVptr['sep_elyte']]))
    #Xk_next = elyte.X
    #phi_elyte_next = elyte.electric_potential

    for j in np.arange(params['N_y_sep'] - 1):
        N_k_in = N_k_out
        i_io_in = i_ext#i_io_out
        phi_elyte = phi_elyte_next
        Xk = Xk_next
        C_k_elyte = C_k_elyte_next

        rho_k = SV[SVptr['sep_elyte']+SV_move+SV_single_sep]
        elyte.TDY = T, abs(sum(rho_k)), rho_k/abs(sum(rho_k))
        Xk_next = elyte.X
        C_k_elyte_next = elyte.concentrations

        C_k_elyte_int = 0.5*(C_k_elyte + C_k_elyte_next)
        D_k_elyte = diff_coeffs(C_k_elyte_int, objs, params)

#        phi_elyte_next = phi_elyte \
#           - (i_io_in + F*ct.gas_constant*T*C_elyte * sum(params['Z_k_elyte']*u_k* \
#           (Xk_next - Xk)*params['dyInv_sep'])) \
#           / (F**2*params['dyInv_sep'] * sum(params['Z_k_elyte']**2 * u_k * C_k_elyte))

        dC_k_elyte = (C_k_elyte_next - C_k_elyte) #* params['dyInv_sep']
        phi_elyte_next = phi_elyte + \
             (-(i_io_out + F*sum(params['Z_k_elyte']*(D_k_elyte[0]*dC_k_elyte / (params['eps_sep_inv']**params['bruggman'])))) / \
             (-F*sum(params['Z_k_elyte']*D_k_elyte[1]*params['dyInv_sep']/(params['eps_sep_inv']**params['bruggman']))))

        phi_elyte_next = phi_elyte - \
            (i_io_out*eps_inv**params['bruggman']/(dyInv*F) + sum(params['Z_k_elyte']*D_k_elyte[0]*dC_k_elyte))/sum(params['Z_k_elyte']*D_k_elyte[1])

#        N_k_out = - u_k * C_k_elyte_elyte * ((ct.gas_constant * T) / Xk_int * (Xk_next - Xk) * params['dyInv_sep'] \
#                  + params['Z_k_elyte'] * F * (phi_elyte_next - phi_elyte) * params['dyInv_sep'])
        N_k_out = -(D_k_elyte[0] * (C_k_elyte_next - C_k_elyte) * params['dyInv_sep'] + \
                   D_k_elyte[1] * (phi_elyte_next - phi_elyte) * params['dyInv_sep']) / \
                   (params['eps_sep_inv']**params['bruggman'])

        i_io_out = F * sum(params['Z_k_elyte'] * N_k_out)

        # Calculate change in electrolyte concentrations
        dSVdt[SVptr['sep_elyte']+SV_move] = (N_k_in - N_k_out) * params['dyInv_sep'] * W_elyte * params['eps_sep_inv']

        # Set potentials and concentrations
        #elyte.TDY = T, sum(SV[SVptr['sep_elyte']+SV_move]), SV[SVptr['sep_elyte']+SV_move]
        #elyte.electric_potential = phi_elyte_next

        # New SV offset value
        SV_move = SV_move + SV_single_sep

    " --- Post-loop --- "
    # Set next 'top' flux as current 'bottom'
    N_k_in = N_k_out
    i_io_in = i_ext#i_io_out
    phi_elyte = phi_elyte_next
    Xk = Xk_next
    C_k_elyte = C_k_elyte_next

    i_io_out = i_ext
    N_k_out = np.zeros_like(N_k_in)

    N_k_out[ptr['Li+']] = i_io_out / (F * params['Z_k_elyte'][ptr['Li+']])
    #print(N_k_out)
    dSVdt[SVptr['sep_elyte']+SV_move] = (N_k_in - N_k_out) * params['dyInv_sep'] * W_elyte * params['eps_sep_inv']"""

    # Set lithium anode potential
#    Li_b.electric_potential = elyte.electric_potential - SV[SVptr['sep_phi']+SV_move]

    # Transport and potential at anode BC
#    N_k_out = Li_s.get_net_production_rates(elyte)
#    print('N_k_out =',N_k_out)
#    i_io_out_check = F * sum(params['Z_k_elyte'] * N_k_out)
#    print('i_io_check =',i_io_out_check,'i_ext = ',i_ext)
#    N_k_out = np.zeros_like(elyte.X)
#    N_k_out[ptr['Li+']] = i_ext * F

    # Calculate double layer potential between separator and anode
#    i_far = Li_s.get_net_production_rates(elyte)[ptr['Li+']] * F
#    i_dl = i_ext - i_far
#    dSVdt[SVptr['sep_phi']+SV_move] = i_dl / (params['C_dl']) #* A_int)

    # Calculate change in electrolyte concentrations
#    dSVdt[SVptr['sep_elyte']+SV_move] = (N_k_in - N_k_out) * params['dyInv_sep'] * params['eps_sep_inv']

    return dSVdt


" ================= HELPER FUNCTIONS ================= "
    
def read_cantera_objs(objs):
    gas = objs['gas']
    ca_bulk = objs['ca_bulk']
    elyte = objs['elyte']
    oxide = objs['oxide']
    ca_surf = objs['ca_surf']
    tpb = objs['tpb']
    air_elyte = objs['air_elyte']
    Li_bulk = objs['Li_bulk']
    Li_surf = objs['Li_surf']

    return gas, ca_bulk, elyte, oxide, ca_surf, tpb, air_elyte, Li_bulk, Li_surf

def read_state(SV, SVptr, j):
    # double layer (i.e. elyte) electric potential:
    phi_elyte = SV[SVptr['phi_dl'][j]]

    # Oxide volume fraction:
    eps_oxide = SV[SVptr['eps_oxide'][j]]

    # Electrolyte species mass fracionts:
    C_k_elyte = SV[SVptr['C_k_elyte_ca'][j]]
    # Remove negative numbers:
    C_k_elyte = 0.5*(C_k_elyte + abs(C_k_elyte) + 1e-24)
    # Remove NaN values:
    C_k_elyte[np.isnan(C_k_elyte)] = 1e-24


    return phi_elyte, eps_oxide, C_k_elyte

def set_states(phi_elyte, C_k_elyte, params, ca_bulk, oxide, elyte, gas):
    
    TP = params['TP']

    oxide.TP = TP
    gas.TP = TP
    elyte.TP = TP

    elyte.X = C_k_elyte/sum(C_k_elyte)

    ca_bulk.electric_potential = 0
    oxide.electric_potential = 0
    elyte.electric_potential = phi_elyte 

    return ca_bulk, oxide, elyte, gas