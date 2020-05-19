"""
Author:
Amy LeBar (20 August 2018)
Li-O2 Battery Model:
This model examines the reactions taking place within the carbon-based
cathode of a Li-O2 battery. Electrolyte = 1 M LiTFSI in TEGDME
"""

# Load any needed modules
import numpy as np
import cantera as ct

import sys
sys.path.append('../')

def LiO2_func(t, SV, params, objs, SVptr):

    if params['transport'] == 'cst':
        from functions.diffusion_coeffs import cst as diff_coeffs
    elif params['transport'] == 'dst':
        from functions.diffusion_coeffs import dst as diff_coeffs
    else:
        raise Exception('Please specify a valid transport model: cst or dst')

    # Pull phases out of 'objs' inside function
    gas = objs['gas']
    cath_b = objs['cath_b']
    elyte = objs['elyte']
    oxide = objs['oxide']
    inter = objs['inter']
    tpb = objs['tpb']
    air_elyte = objs['air_elyte']
    Li_b = objs['Li_b']
    Li_s = objs['Li_s']
    
    # Read out and store some constants:
    # Faraday's constant (C/kmol e-)
    F = ct.faraday
    # oxide molar volume
    Vbar_oxide = oxide.mean_molecular_weight / oxide.density_mass       
    # ELectrolyte molar weights
    W_elyte = elyte.molecular_weights

    dSVdt = np.zeros_like(SV)

    # Pull parameters out of 'params' inside function
    i_ext = params['i_ext']
    T = params['TP'][0]

    N_y_cath = params['N_y_cath']
    dyInv_cath = params['dyInv_cath']

    N_y_sep = params['N_y_sep']
    dyInv_sep = params['dyInv_sep']
    
    E_sep_inv = params['E_sep_inv']

    C_dl = params['C_dl']

    Zk_elyte = params['Zk_elyte']

    W_elyte = elyte.molecular_weights

    " ======================================== CATHODE ======================================== "
    " --- Pre-loop --- "
    j = 0
    # Set potentials and concentrations for 'next'
    cath_b.electric_potential = 0.
    phi_elyte_next = SV[SVptr['phi_dl'][j]]
    elyte.electric_potential = phi_elyte_next
    rho = abs(sum(SV[SVptr['rho_k_elyte'][j]]))
    elyte.TDY = T, rho, SV[SVptr['rho_k_elyte'][j]]/rho
    Xk_next = elyte.X
    Ck_next = elyte.concentrations
    #inter.coverages = abs(SV[SVptr['theta']])

    E_oxide_next = SV[SVptr['eps_oxide'][j]]

    E_elyte_next = 1. - params['eps_carbon'] - E_oxide_next

    # Mass transport and ionic current at air/elyte BC
    Nk_top = air_elyte.get_net_production_rates(elyte)
    i_io_top = 0.

    Nk_bot = np.zeros_like(Nk_top)

    for j in np.arange(N_y_cath - 1):
        phi_elyte_this = phi_elyte_next
        phi_elyte_next = SV[SVptr['phi_dl'][j+1]]
        Xk_this = Xk_next
        Ck_this = Ck_next
        E_elyte_this = E_elyte_next
        E_oxide_this = E_oxide_next

        # Area of oxide surface per unit volume:
        A_oxide = 3.*E_oxide_this/params['r_oxide']

        # Area of carbon surface per unit volume:
        A_carb = params['A_carbon_init'] - E_oxide_this*1.5/params['r_oxide']

        # Calculate chemical production terms:
        i_far = tpb.get_net_production_rates(cath_b) * F * A_carb
        sdot_elyte_int = tpb.get_net_production_rates(elyte) * A_carb
        sdot_elyte_dl = np.zeros_like(sdot_elyte_int)
        sdot_oxide = tpb.get_net_production_rates(oxide) * A_carb

        # Elyte state for 'next node'
        rho = abs(sum(SV[SVptr['rho_k_elyte'][j+1]]))
        elyte.TDY = T, rho, abs(SV[SVptr['rho_k_elyte'][j+1]])/rho
        Xk_next = elyte.X
        Ck_next = elyte.concentrations

        # Concentration at interface between nodes:
        Ck_int = 0.5*(Ck_this + Ck_next)

        Dk_elyte = diff_coeffs(Ck_int, objs, params)

        Nk_bot = -(Dk_elyte[0] * (Ck_next - Ck_this) + \
            Dk_elyte[1] * (phi_elyte_next - phi_elyte_this)) * dyInv_cath * \
            E_elyte_this**params['bruggman']

        # Ionic current out of node
        i_io_bot = F * sum(Zk_elyte * Nk_bot)

        #elyte.TDY = T, sum(SV[SVptr['rho_k_elyte'][j]]), SV[SVptr['rho_k_elyte'][j]]

        # Calculate change in double layer potential
        i_dl = (i_io_top - i_io_bot) * dyInv_cath + i_far#*A_int          # double layer current
        dSVdt[SVptr['phi_dl'][j]] = i_dl / (C_dl*A_carb)

        # Calculate change in electrolyte concentrations
        sdot_elyte_dl[params['i_dl_species']] = -i_dl / \
            (F * Zk_elyte[params['i_dl_species']])
        
        dSVdt[SVptr['rho_k_elyte'][j]] = ((Nk_top - Nk_bot) * dyInv_cath \
            + (sdot_elyte_int + sdot_elyte_dl)) * W_elyte / E_elyte_this

        # Calculate change in oxide volume fraction
        dSVdt[SVptr['eps_oxide'][j]] = Vbar_oxide*sdot_oxide

        # Set next 'top' flux and ionic current as current 'bottom'
        Nk_top = Nk_bot
        i_io_top = i_io_bot

        # Set potentials and concentrations for 'next' node
        cath_b.electric_potential = 0.
        elyte.electric_potential = phi_elyte_next
        E_oxide_next = SV[SVptr['eps_oxide'][j+1]]
        E_elyte_next = 1. - params['eps_carbon'] - E_oxide_next

    " --- Post-loop --- "
    # Set previous 'next' as current 'this'
    j = N_y_cath - 1
    # Read out reaction terms, before setting electrolyte object state to that
    #   of the first separator node:

    phi_elyte_this = phi_elyte_next
    Xk_this = Xk_next
    Ck_this = Ck_next
    E_elyte_this = E_elyte_next
    E_oxide_this = E_oxide_next

    # Area of oxide surface per unit volume:
    A_oxide = 3.*E_oxide_this/params['r_oxide']

    # Area of carbon surface per unit volume:
    A_carb = params['A_carbon_init'] - E_oxide_this*1.5/params['r_oxide']

    # Calculate Faradaic current
    i_far = tpb.get_net_production_rates(cath_b) * A_carb * F

    # Production rate of electrolyte species:
    sdot_elyte_int = tpb.get_net_production_rates(elyte) * A_carb
    sdot_elyte_dl = np.zeros_like(sdot_elyte_int)
    sdot_oxide = tpb.get_net_production_rates(oxide) * A_carb

    i_io_bot = i_ext

    Nk_bot = np.zeros_like(Nk_top)
    Nk_bot[params['i_dl_species']] = i_io_bot / \
        (F * Zk_elyte[params['i_dl_species']])

    # Calculate change in double layer potential
    i_dl = (i_io_top - i_io_bot) * dyInv_cath + i_far#*A_carb              # double layer current
    dSVdt[SVptr['phi_dl'][j]] = i_dl / (C_dl*A_carb)

    # Calculate change in electrolyte concentrations
    sdot_elyte_dl[params['i_dl_species']] = -i_dl \
        / (F * Zk_elyte[params['i_dl_species']])
    dSVdt[SVptr['rho_k_elyte'][j]] = ((Nk_top - Nk_bot) * dyInv_cath + \
                                    (sdot_elyte_int + sdot_elyte_dl)) * W_elyte / E_elyte_this

    dSVdt[SVptr['eps_oxide'][j]] = Vbar_oxide*sdot_oxide


    " ======================================== SEPARATOR ======================================== "
    # Initialize SV pointer offset for separator
    """

    " --- Pre-loop --- "
    elyte.TDY = T, abs(sum(SV[SVptr['sep_elyte']])), SV[SVptr['sep_elyte']]/abs(sum(SV[SVptr['sep_elyte']]))
    #Xk_next = elyte.X
    #phi_elyte_next = elyte.electric_potential

    for j in np.arange(N_y_sep - 1):
        Nk_top = Nk_bot
        i_io_top = i_ext#i_io_bot
        phi_elyte_this = phi_elyte_next
        Xk_this = Xk_next
        Ck_this = Ck_next

        rho_k = SV[SVptr['sep_elyte']+SV_move+SV_single_sep]
        elyte.TDY = T, abs(sum(rho_k)), rho_k/abs(sum(rho_k))
        Xk_next = elyte.X
        Ck_next = elyte.concentrations

        Ck_int = 0.5*(Ck_this + Ck_next)
        Dk_elyte = diff_coeffs(Ck_int, objs, params)

#        phi_elyte_next = phi_elyte_this - (i_io_top + F*ct.gas_constant*T*C_elyte \
#                        * sum(Zk_elyte*u_k*(Xk_next - Xk_this)*dyInv_sep)) \
#                        / (F**2*dyInv_sep * sum(Zk_elyte**2 * u_k * Ck_elyte))

        dCk = (Ck_next - Ck_this) #* dyInv_sep
        phi_elyte_next = phi_elyte_this + \
             (-(i_io_bot + F*sum(Zk_elyte*(Dk_elyte[0]*dCk / (E_sep_inv**params['bruggman'])))) / \
             (-F*sum(Zk_elyte*Dk_elyte[1]*dyInv_sep/(E_sep_inv**params['bruggman']))))

        phi_elyte_next = phi_elyte_this - \
            (i_io_bot*E_inv**params['bruggman']/(dyInv*F) + sum(Zk_elyte*Dk_elyte[0]*dCk))/sum(Zk_elyte*Dk_elyte[1])

#        Nk_bot = - u_k * Ck_elyte * ((ct.gas_constant * T) / Xk_int * (Xk_next - Xk_this) * dyInv_sep \
#                  + Zk_elyte * F * (phi_elyte_next - phi_elyte_this) * dyInv_sep)
        Nk_bot = -(Dk_elyte[0] * (Ck_next - Ck_this) * dyInv_sep + \
                   Dk_elyte[1] * (phi_elyte_next - phi_elyte_this) * dyInv_sep) / \
                   (E_sep_inv**params['bruggman'])

        i_io_bot = F * sum(Zk_elyte * Nk_bot)

        # Calculate change in electrolyte concentrations
        dSVdt[SVptr['sep_elyte']+SV_move] = (Nk_top - Nk_bot) * dyInv_sep * W_elyte * E_sep_inv

        # Set potentials and concentrations
        #elyte.TDY = T, sum(SV[SVptr['sep_elyte']+SV_move]), SV[SVptr['sep_elyte']+SV_move]
        #elyte.electric_potential = phi_elyte_next

        # New SV offset value
        SV_move = SV_move + SV_single_sep

    " --- Post-loop --- "
    # Set next 'top' flux as current 'bottom'
    Nk_top = Nk_bot
    i_io_top = i_ext#i_io_bot
    phi_elyte_this = phi_elyte_next
    Xk_this = Xk_next
    Ck_this = Ck_next

    i_io_bot = i_ext
    Nk_bot = np.zeros_like(Nk_top)

    Nk_bot[ptr['Li+']] = i_io_bot / (F * Zk_elyte[ptr['Li+']])
    #print(Nk_bot)
    dSVdt[SVptr['sep_elyte']+SV_move] = (Nk_top - Nk_bot) * dyInv_sep * W_elyte * E_sep_inv"""

    # Set lithium anode potential
#    Li_b.electric_potential = elyte.electric_potential - SV[SVptr['sep_phi']+SV_move]

    # Transport and potential at anode BC
#    Nk_bot = Li_s.get_net_production_rates(elyte)
#    print('Nk_bot =',Nk_bot)
#    i_io_bot_check = F * sum(Zk_elyte * Nk_bot)
#    print('i_io_check =',i_io_bot_check,'i_ext = ',i_ext)
#    Nk_bot = np.zeros_like(elyte.X)
#    Nk_bot[ptr['Li+']] = i_ext * F

    # Calculate double layer potential between separator and anode
#    i_far = Li_s.get_net_production_rates(elyte)[ptr['Li+']] * F
#    i_dl = i_ext - i_far
#    dSVdt[SVptr['sep_phi']+SV_move] = i_dl / (C_dl) #* A_int)

    # Calculate change in electrolyte concentrations
#    dSVdt[SVptr['sep_elyte']+SV_move] = (Nk_top - Nk_bot) * dyInv_sep * E_sep_inv

    return dSVdt
