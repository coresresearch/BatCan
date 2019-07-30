"""
Author:
Amy LeBar 24 April 2019

Diffusion coefficient function
Inputs: bulk diffusion coefficents,

Outputs: chemical diffusion coefficient (Dk_elyte), and migration term (Dk_mig_elyte)
"""
# Load needed modules
import cantera as ct
import numpy as np

# Dilute Solution Theory (dst)
def dst(Ck_elyte,objs,params):
    Dk_elyte = params['Dk_elyte_o']
    Dk_mig_elyte = params['Dk_mig_elyte_o'] * Ck_elyte

    return Dk_elyte, Dk_mig_elyte

# Concentrated Solution Theory (cst)
def cst(Ck_elyte,objs,params):
    elyte = objs['elyte']

    C_Li = Ck_elyte[params['iLi_elyte']]*1000
    C_EC = Ck_elyte[params['iEC_elyte']]*1000
    C_EMC = Ck_elyte[params['iEMC_elyte']]*1000

    Dk_elyte = params['Dk_elyte_o']

    # Chemical diffusion coefficent for Li+ and PF6- in elyte
    Dk_elyte[params['iLi_elyte']] = (params['D_Li_a']*C_Li**2 + \
                                params['D_Li_b']*C_Li + params['D_Li_c']) * \
                                (params['D_Li_d']*C_Li**2 + params['D_Li_e']*C_Li + \
                                params['D_Li_f']) / (params['D_Li_g'] + params['D_Li_h']*C_Li + \
                                params['D_Li_i']*C_Li**2 + params['D_Li_j']*C_Li**3) * \
                                sum(1000*Ck_elyte) / (C_EC + C_EMC)

    Dk_elyte[params['iPF6_elyte']] = Dk_elyte[params['iLi_elyte']]

    # Positive ion transference number
    tk_elyte_o = params['t_elyte_a'] + params['t_elyte_b']*C_Li + \
                  params['t_elyte_c']*C_Li**2 + params['t_elyte_d']*C_Li**3

    tk_elyte = abs(params['polarity_k']) * \
            ((params['polarity_k'] + 1) * tk_elyte_o/2 - \
            (params['polarity_k'] - 1) * (1 - tk_elyte_o)/2)

    # Effective conductivity
    sigma = params['sigma_elyte_a']*C_Li + \
            params['sigma_elyte_b']*C_Li**1.5 + \
            params['sigma_elyte_c']*C_Li**3

    # Thermodynamic factor - (1 - df/d(ln C_Li))
    thermo_factor = (params['gamma_elyte_a']*C_Li**2 + \
            params['gamma_elyte_b']*C_Li + params['gamma_elyte_c']) / \
            (params['gamma_elyte_d'] + params['gamma_elyte_e']*C_Li + \
            params['gamma_elyte_f']*C_Li**2 + params['gamma_elyte_g']*C_Li**3)

    # Diffusional conductivity
    sigma_D = 2*ct.gas_constant*elyte.TP[0] * \
              sigma*(tk_elyte_o - 1)*thermo_factor / ct.faraday

    Dk_elyte = Dk_elyte + sigma_D*tk_elyte / (Ck_elyte*params['divCharge']*ct.faraday)

    Dk_mig_elyte = tk_elyte * sigma / params['divCharge'] / ct.faraday

    return Dk_elyte, Dk_mig_elyte
