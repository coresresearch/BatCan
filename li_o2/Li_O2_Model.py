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

def LiO2_func(t,SV,params,objs,geom,ptr,SVptr):

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
    air_elyte = objs['air_elyte']
    Li_b = objs['Li_b']
    Li_s = objs['Li_s']

    dSVdt = np.zeros_like(SV)

    # Pull parameters out of 'params' inside function
    i_ext = params['i_ext']
    T = params['T']

    Ny_cath = params['Ny_cath']
    dyInv_cath = params['dyInv_cath']

    Ny_sep = params['Ny_sep']
    dyInv_sep = params['dyInv_sep']

    E_carb_inv = params['E_carb_inv']
    E_sep_inv = params['E_sep_inv']

    A_int = params['A_int']
    C_dl = params['C_dl']
    R_sep = params['R_sep']

    Zk_elyte = params['Zk_elyte']

    # Number of variables in each node
    SV_single_cath = params['SV_single_cath']
    SV_single_sep = params['SV_single_sep']

    W_elyte = elyte.molecular_weights

    " ======================================== CATHODE ======================================== "
    " --- Pre-loop --- "
    # Set potentials and concentrations for 'next'
    cath_b.electric_potential = 0.
    phi_elyte_next = SV[SVptr['phi']]
    elyte.electric_potential = phi_elyte_next
    rho = abs(sum(SV[SVptr['elyte']]))
    elyte.TDY = T, rho, SV[SVptr['elyte']]/rho
    Xk_next = elyte.X
    Ck_next = elyte.concentrations
    inter.coverages = abs(SV[SVptr['theta']])

    # Mass transport and ionic current at air/elyte BC
    Nk_top = air_elyte.get_net_production_rates(elyte)
    i_io_top = 0.

    Nk_bot = np.zeros_like(Nk_top)

    # Initialize SV pointer offset
    SV_move = 0

    for j in np.arange(Ny_cath - 1):
        phi_elyte_this = phi_elyte_next
        phi_elyte_next = SV[SVptr['phi']+SV_move+SV_single_cath]
        Xk_this = Xk_next
        Ck_this = Ck_next

        # Calculate chemical production terms:
        i_far = inter.get_net_production_rates(cath_b) * ct.faraday
        sdot_elyte_int = inter.get_net_production_rates(elyte) * A_int
        sdot_elyte_dl = np.zeros_like(sdot_elyte_int)
        sdot_surf = inter.get_net_production_rates(inter)

        # Elyte state for 'next node'
        rho = abs(sum(SV[SVptr['elyte']+SV_move+SV_single_cath]))
        elyte.TDY = T, rho, abs(SV[SVptr['elyte']+SV_move+SV_single_cath])/rho
        Xk_next = elyte.X
        Ck_next = elyte.concentrations

        # Concentration at interface between nodes:
        Ck_int = 0.5*(Ck_this + Ck_next)

        Dk_elyte = diff_coeffs(Ck_int, objs, params)

        Nk_bot = -(Dk_elyte[0] * (Ck_next - Ck_this) + \
            Dk_elyte[1] * (phi_elyte_next - phi_elyte_this)) * dyInv_cath / \
            E_carb_inv**geom['bruggman']

        #print(j,Nk_bot)

        # Ionic current out of node
        i_io_bot = ct.faraday * sum(Zk_elyte * Nk_bot)

        #elyte.TDY = T, sum(SV[SVptr['elyte']+SV_move]), SV[SVptr['elyte']+SV_move]

        # Calculate change in double layer potential
        i_dl = (i_io_top - i_io_bot) * dyInv_cath + i_far*A_int          # double layer current
        dSVdt[SVptr['phi']+SV_move] = i_dl / (C_dl*A_int)

        # Calculate change in electrolyte concentrations
        sdot_elyte_dl[ptr['Li+']] = -i_dl / (ct.faraday * Zk_elyte[ptr['Li+']])
        dSVdt[SVptr['elyte']+SV_move] = ((Nk_top - Nk_bot) * dyInv_cath \
                                        + (sdot_elyte_int + sdot_elyte_dl)) * W_elyte * E_carb_inv

        # Calculate change in surface coverages
        dSVdt[SVptr['theta']+SV_move] = sdot_surf/inter.site_density

        # Set next 'top' flux and ionic current as current 'bottom'
        Nk_top = Nk_bot
        i_io_top = i_io_bot

        SV_move = SV_move + SV_single_cath             # new SV offset value

        # Set potentials and concentrations for 'next' node
        cath_b.electric_potential = 0.
        #elyte.TDY = T, sum(SV[SVptr['elyte']+SV_move]), SV[SVptr['elyte']+SV_move]
        coverages = abs(SV[SVptr['theta']+SV_move])
        inter.coverages = coverages#SV[SVptr['theta']+SV_move]
        elyte.electric_potential = phi_elyte_next

    " --- Post-loop --- "
    # Set previous 'next' as current 'this'

    # Read out reaction terms, before setting electrolyte object state to that
    #   of the first separator node:

    # Calculate Faradaic current
    i_far = inter.get_net_production_rates(cath_b) * ct.faraday

    # Production rate of electrolyte species:
    sdot_elyte_int = inter.get_net_production_rates(elyte) * A_int
    sdot_elyte_dl = np.zeros_like(sdot_elyte_int)

    # Production rate of surface species:
    sdot_surf = inter.get_net_production_rates(inter)

    phi_elyte_this = phi_elyte_next
    Xk_this = Xk_next
    Ck_this = Ck_next

    elyte.TDY = T, abs(sum(SV[SVptr['sep_elyte']])), SV[SVptr['sep_elyte']]/abs(sum(SV[SVptr['sep_elyte']]))
    Xk_next = elyte.X
    Ck_next = elyte.concentrations

    elyte.TDY = T, abs(sum(SV[SVptr['elyte']+SV_move])), SV[SVptr['elyte']+SV_move]/abs(sum(SV[SVptr['elyte']+SV_move]))

    # Molar transport and ionic current at separator BC
    i_io_bot = i_ext

    """Ck_int = (Ck_this/dyInv_cath + Ck_next/dyInv_sep)/(1./dyInv_cath + 1./dyInv_sep)

    Dk_elyte = diff_coeffs(Ck_int, objs, params)

    dyInv = 2./(1./dyInv_cath + 1./dyInv_sep)
    E_inv = (1./dyInv_cath + 1./dyInv_sep)/(1./(E_carb_inv*dyInv_cath) + 1./(E_sep_inv*dyInv_sep))

    dCk = (Ck_next - Ck_this)# * dyInv
#    phi_elyte_next = phi_elyte_this + \
#         (-(i_io_bot + ct.faraday*sum(Zk_elyte*(Dk_elyte[0]*dCk / (E_inv**geom['bruggman'])))) / \
#         (-ct.faraday*sum(Zk_elyte*Dk_elyte[1]*dyInv / (E_inv**geom['bruggman']))))

    phi_elyte_next = phi_elyte_this - \
        (i_io_bot*E_inv**geom['bruggman']/(dyInv*ct.faraday) + sum(Zk_elyte*Dk_elyte[0]*dCk))/sum(Zk_elyte*Dk_elyte[1])


    Nk_bot = -(Dk_elyte[0] * (Ck_next - Ck_this) + \
            Dk_elyte[1] * (phi_elyte_next - phi_elyte_this))* dyInv / \
            (E_inv**geom['bruggman'])

    print(Nk_bot)

    i_io_bot_check = ct.faraday * sum(Zk_elyte * Nk_bot)"""

    Nk_bot = np.zeros_like(Nk_top)
    Nk_bot[ptr['Li+']] = i_io_bot / (ct.faraday * Zk_elyte[ptr['Li+']])

    # Calculate change in double layer potential
    i_dl = (i_io_top - i_io_bot) * dyInv_cath + i_far*A_int              # double layer current
    dSVdt[SVptr['phi']+SV_move] = i_dl / (C_dl*A_int)

    # Calculate change in electrolyte concentrations
    sdot_elyte_dl[ptr['Li+']] = -i_dl / (ct.faraday * Zk_elyte[ptr['Li+']])
    dSVdt[SVptr['elyte']+SV_move] = ((Nk_top - Nk_bot) * dyInv_cath + \
                                    (sdot_elyte_int + sdot_elyte_dl)) * W_elyte * E_carb_inv

    # Calculate change in surface coverages
    dSVdt[SVptr['theta']+SV_move] = sdot_surf/ inter.site_density

    " ======================================== SEPARATOR ======================================== "
    # Initialize SV pointer offset for separator
    """SV_move = 0

    " --- Pre-loop --- "
    elyte.TDY = T, abs(sum(SV[SVptr['sep_elyte']])), SV[SVptr['sep_elyte']]/abs(sum(SV[SVptr['sep_elyte']]))
    #Xk_next = elyte.X
    #phi_elyte_next = elyte.electric_potential

    for j in np.arange(Ny_sep - 1):
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

#        phi_elyte_next = phi_elyte_this - (i_io_top + ct.faraday*ct.gas_constant*T*C_elyte \
#                        * sum(Zk_elyte*u_k*(Xk_next - Xk_this)*dyInv_sep)) \
#                        / (ct.faraday**2*dyInv_sep * sum(Zk_elyte**2 * u_k * Ck_elyte))

        dCk = (Ck_next - Ck_this) #* dyInv_sep
        phi_elyte_next = phi_elyte_this + \
             (-(i_io_bot + ct.faraday*sum(Zk_elyte*(Dk_elyte[0]*dCk / (E_sep_inv**geom['bruggman'])))) / \
             (-ct.faraday*sum(Zk_elyte*Dk_elyte[1]*dyInv_sep/(E_sep_inv**geom['bruggman']))))

        phi_elyte_next = phi_elyte_this - \
            (i_io_bot*E_inv**geom['bruggman']/(dyInv*ct.faraday) + sum(Zk_elyte*Dk_elyte[0]*dCk))/sum(Zk_elyte*Dk_elyte[1])

#        Nk_bot = - u_k * Ck_elyte * ((ct.gas_constant * T) / Xk_int * (Xk_next - Xk_this) * dyInv_sep \
#                  + Zk_elyte * ct.faraday * (phi_elyte_next - phi_elyte_this) * dyInv_sep)
        Nk_bot = -(Dk_elyte[0] * (Ck_next - Ck_this) * dyInv_sep + \
                   Dk_elyte[1] * (phi_elyte_next - phi_elyte_this) * dyInv_sep) / \
                   (E_sep_inv**geom['bruggman'])

        i_io_bot = ct.faraday * sum(Zk_elyte * Nk_bot)

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

    Nk_bot[ptr['Li+']] = i_io_bot / (ct.faraday * Zk_elyte[ptr['Li+']])
    #print(Nk_bot)
    dSVdt[SVptr['sep_elyte']+SV_move] = (Nk_top - Nk_bot) * dyInv_sep * W_elyte * E_sep_inv"""

    # Set lithium anode potential
#    Li_b.electric_potential = elyte.electric_potential - SV[SVptr['sep_phi']+SV_move]

    # Transport and potential at anode BC
#    Nk_bot = Li_s.get_net_production_rates(elyte)
#    print('Nk_bot =',Nk_bot)
#    i_io_bot_check = ct.faraday * sum(Zk_elyte * Nk_bot)
#    print('i_io_check =',i_io_bot_check,'i_ext = ',i_ext)
#    Nk_bot = np.zeros_like(elyte.X)
#    Nk_bot[ptr['Li+']] = i_ext * ct.faraday

    # Calculate double layer potential between separator and anode
#    i_far = Li_s.get_net_production_rates(elyte)[ptr['Li+']] * ct.faraday
#    i_dl = i_ext - i_far
#    dSVdt[SVptr['sep_phi']+SV_move] = i_dl / (C_dl) #* A_int)

    # Calculate change in electrolyte concentrations
#    dSVdt[SVptr['sep_elyte']+SV_move] = (Nk_top - Nk_bot) * dyInv_sep * E_sep_inv

    return dSVdt
