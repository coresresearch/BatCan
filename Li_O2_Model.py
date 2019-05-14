
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
from diffusion_coeffs import dst as dst
from diffusion_coeffs import cst as cst

def LiO2_func(t,SV,params,objs,geom,ptr,SVptr):
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
    dRhoElytedt = np.zeros_like(SVptr['elyte'])
    dThetadt = np.zeros_like(SVptr['theta'])

    # Pull parameters out of 'params' inside function
    dyInv = params['dyInv']
    Ny = params['Ny']
    A_int = params['A_int']
    C_dl = params['C_dl']
    SV_single = params['SV_single']
    E_carb_inv = params['E_carb_inv']
    T = params['T']
    Zk_elyte = params['Zk_elyte']

    W_elyte = elyte.molecular_weights

    " --- Pre-loop --- "
    # Set potentials and concentrations for 'next'
    cath_b.electric_potential = 0
    elyte.electric_potential = SV[SVptr['phi']]
    Yk_next = SV[SVptr['elyte']]
    elyte.TDY = T, sum(SV[SVptr['elyte']]), Yk_next
    inter.coverages = SV[SVptr['theta']]
    phi_elyte_next = elyte.electric_potential

    # Mass transport and ionic current at air/elyte BC
    Nk_top = air_elyte.get_net_production_rates(elyte)
    i_io_top = 0

    Nk_bot = np.zeros_like(Nk_top)

    # Initialize SV pointer offset
    SV_move = 0

    for j in np.arange(Ny - 1):
        phi_elyte_this = phi_elyte_next
        phi_elyte_next = SV[SVptr['phi']+SV_move+SV_single]
        Yk_this = Yk_next
        Yk_next = SV[SVptr['elyte']+SV_move+SV_single]

        # Mass transport and ionic current
        u_k = np.zeros_like(Yk_next)
        Ck_elyte = elyte.concentrations
        
        if params['transport'] == 'cst':
            Dk_elyte = cst(Ck_elyte,objs,params)[0]
            Dk_elyte_mig = cst(Ck_elyte,objs,params)[1]
        if params['transport'] == 'dst':
            Dk_elyte = dst(Ck_elyte,params)[0]
            Dk_elyte_mig = dst(Ck_elyte,params)[1]

        u_k = Dk_elyte #/ (ct.gas_constant * T)
        grad_mu_k = 
        grad_mu_k = (Yk_next - Yk_this) * dyInv + (Zk_elyte * ct.faraday * (phi_elyte_next - phi_elyte_this) * dyInv)
#        Nk_bot = -u_k * Ck_elyte * grad_mu_k

        Nk_bot = - (Dk_elyte * (Yk_next - Yk_this) / Xk_int + Dk_elyte_mig * (phi_elyte_next - phi_elyte_this)) * dyInv # * params['E_elyte_0']**geom['bruggman']

        i_io_bot = ct.faraday * sum(Zk_elyte * Nk_bot)

        # Calculate Faradaic current
        i_far = inter.get_net_production_rates(cath_b) * ct.faraday

        # Calculate change in double layer potential
        i_dl = (i_io_top - i_io_bot) * dyInv + i_far*A_int          # double layer current
        dPhidt = i_dl / (C_dl*A_int)                                # double layer potential

        # Calculate change in electrolyte concentrations
        sdot_int = inter.get_net_production_rates(elyte) * A_int
        sdot_dl = np.zeros_like(sdot_int)
        sdot_dl[ptr['Li+']] = i_dl / (ct.faraday * Zk_elyte[ptr['Li+']])
        dRhoElytedt = ((Nk_top - Nk_bot) * dyInv + (sdot_int + sdot_dl)) * W_elyte * E_carb_inv

        # Calculate change in surface coverages
        dThetadt = inter.get_net_production_rates(inter) / inter.site_density

        # Load differential eqns into dSVdt
        dSVdt[SVptr['phi']+SV_move] = dPhidt                        # double layer potential
        dSVdt[SVptr['elyte']+SV_move] = dRhoElytedt                 # electrolyte concentration
        dSVdt[SVptr['theta']+SV_move] = dThetadt                    # particle surface coverages

        # Set next 'top' flux and ionic current as this 'bottom'
        Nk_top = Nk_bot
        i_io_top = i_io_bot

        SV_move = SV_move + SV_single                               # new SV offset value

        # Set potentials and concentrations
        cath_b.electric_potential = 0
        elyte.TDY = T, sum(SV[SVptr['elyte']+SV_move]), SV[SVptr['elyte']+SV_move]
        inter.coverages = SV[SVptr['theta']+SV_move]
        elyte.electric_potential = phi_elyte_next

    " --- Post-loop --- "
    # Mass transport and ionic current at separator BC
    i_io_bot = params['i_ext']
    Nk_bot = np.zeros(elyte.n_species)
    Nk_bot[ptr['Li+']] = i_io_bot / (ct.faraday * Zk_elyte[ptr['Li+']])

    # Calculate Faradaic current
    i_far = inter.get_net_production_rates(cath_b) * ct.faraday

    # Calculate change in double layer potential
    i_dl = (i_io_top - i_io_bot) * dyInv + i_far*A_int              # double layer current
    dPhidt = i_dl / (C_dl*A_int)                                    # double layer potential

    # Calculate change in electrolyte concentrations
    sdot_int = inter.get_net_production_rates(elyte) * A_int
    sdot_dl = np.zeros_like(sdot_int)
    sdot_dl[ptr['Li+']] = i_dl / (ct.faraday * Zk_elyte[ptr['Li+']])
    dRhoElytedt = ((Nk_top - Nk_bot) * dyInv + (sdot_int + sdot_dl)) * W_elyte * E_carb_inv

    # Calculate change in surface coverages
    dThetadt = inter.get_net_production_rates(inter) / inter.site_density

    # Load differentials into dSVdt
    dSVdt[SVptr['phi']+SV_move] = dPhidt                        # double layer potential
    dSVdt[SVptr['elyte']+SV_move] = dRhoElytedt                 # electrolyte concentration
    dSVdt[SVptr['theta']+SV_move] = dThetadt                    # particle surface coverages

    return dSVdt
