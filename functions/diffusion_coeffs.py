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

    # Species concentrations, converted to mol/m3:
    C_Li = Ck_elyte[params['i_Li_elyte']]*1000
    C_solvent = 0.
    for i_sp in params['i_solvent_elyte']:
        C_solvent += Ck_elyte[i_sp]*1000
        #     C_EC = Ck_elyte[params['iEC_elyte']]*1000
        #     C_EMC = Ck_elyte[params['iEMC_elyte']]*1000

    Dk_elyte = params['Dk_elyte_o']

    # Chemical diffusion coefficent for Li+ and PF6- in elyte
    Dk_elyte[params['i_Li_elyte']] = (params['D_Li_CST'][0]*C_Li**2 \
        + params['D_Li_CST'][1]*C_Li + params['D_Li_CST'][2]) \
        * (params['D_Li_CST'][3]*C_Li**2 + params['D_Li_CST'][4]*C_Li  \
        + params['D_Li_CST'][5]) / (params['D_Li_CST'][6] \
        + params['D_Li_CST'][7]*C_Li + params['D_Li_CST'][8]*C_Li**2 \
        + params['D_Li_CST'][9]*C_Li**3) * sum(1000*Ck_elyte) / C_solvent

    Dk_elyte[params['i_counter_ion']] = Dk_elyte[params['i_Li_elyte']]

    # Positive ion transference number
    tk_elyte_o = params['t_elyte'][0] + params['t_elyte'][1]*C_Li \
                + params['t_elyte'][2]*C_Li**2 + params['t_elyte'][3]*C_Li**3

    tk_elyte = abs(params['polarity_k']) * \
            ((params['polarity_k'] + 1) * tk_elyte_o/2 - \
            (params['polarity_k'] - 1) * (1 - tk_elyte_o)/2)

    # Effective conductivity
    sigma = params['sigma_elyte'][0]*C_Li + \
            params['sigma_elyte'][1]*C_Li**1.5 + \
            params['sigma_elyte'][2]*C_Li**3

    # Thermodynamic factor - (1 - df/d(ln C_Li))
    thermo_factor = (params['gamma_elyte'][0]*C_Li**2 + \
            params['gamma_elyte'][1]*C_Li + params['gamma_elyte'][2]) / \
            (params['gamma_elyte'][3] + params['gamma_elyte'][4]*C_Li + \
            params['gamma_elyte'][5]*C_Li**2 + params['gamma_elyte'][6]*C_Li**3)

    # Diffusional conductivity
    sigma_D = 2*ct.gas_constant*elyte.TP[0] * \
        sigma*(tk_elyte_o - 1)*thermo_factor / ct.faraday

    Dk_elyte = Dk_elyte + sigma_D * tk_elyte * params['div_charge'] \
            / (Ck_elyte*ct.faraday)

    Dk_mig_elyte = tk_elyte * sigma * params['div_charge'] / ct.faraday

    return Dk_elyte, Dk_mig_elyte
