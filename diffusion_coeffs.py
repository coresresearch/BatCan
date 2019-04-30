"""
Author:
Amy LeBar 24 April 2019

Diffusion coefficient function
Inputs:

Outputs: Dk_elyte, Dk_mig_elyte
"""
# Load needed modules
import cantera as ct

params = {}

D_Li_elyte = 4e-10                      # bulk diff coeff Li+ in elyte (m2/s)
D_PF6_elyte = 4e-10                     # bulk diff coeff PF6- in elyte (m2/s)
D_LiPF6_elyte = 4e-10                   # bulk diff coeff LiPF6 in elyte (m2/s)
D_O2_elyte = 7e-10                      # bulk diff coeff O2 in elyte (m2/s)
D_EC_elyte = 1                          # EC diffusion is fast
D_EMC_elyte = 1                         # EMC diffusion is fast

# CST Parameters:
params['D_Li_a'] = 8.794e-17
params['D_Li_b'] = -3.972e-13
params['D_Li_c'] = 4.862e-10
params['D_Li_d'] = 0.28687e-6
params['D_Li_e'] = 0.74678e-3
params['D_Li_f'] = 0.44130
params['D_Li_g'] = 0.5508
params['D_Li_h'] = 0.4717e-3
params['D_Li_i'] = -0.4106e-6
params['D_Li_j'] = 0.1287e-9
params['D_Li_k'] = 2.0

# Electrolyte conductivity (S/m)
params['sigma_elyte_a'] = 3.329e-3
params['sigma_elyte_b'] = -7.9373e-5
params['sigma_elyte_c'] = 0.1297e-9

# Liquid activity coefficient
params['gamma_elyte_a'] = 2.8687e-7
params['gamma_elyte_b'] = 7.4678e-4
params['gamma_elyte_c'] = 0.44130
params['gamma_elyte_d'] = 0.5508
params['gamma_elyte_e'] = 4.717e-4
params['gamma_elyte_f'] = -4.106e-7
params['gamma_elyte_g'] = 1.287e-10

params['t_elyte_a'] = 0.4492
params['t_elyte_b'] = -4.717e-4
params['t_elyte_c'] = 4.106e-7
params['t_elyte_d'] = -1.287e-10

params['iO2_elyte'] = 0
params['iLi_elyte'] = 1
params['iPF6_elyte'] = 2
params['iEC_elyte'] = 3
params['iEMC_elyte'] = 4

# Dilute Solution Theory (dst)
def dst(Dk_elyte,Dk_mig_elyte,Ck_elyte,params):
    Dk_elyte = params['Dk_elyte_o']
    Dk_mig_elyte = params['Dk_mig_elyte_o'] * Ck_elyte

# Concentrated Solution Theory (cst)
def cst(Dk_elyte,Dk_mig_elyte,Ck_elyte,objs,params):
    elyte = objs['elyte']

    C_Li = Ck_elyte[params['iLi_elyte']]*1000
    C_EC = Ck_elyte[params['iEC_elyte']]*1000
    C_EMC = Ck_elyte[params['iEMC_elyte']]*1000

    Dk_elyte = params.Dk_elyte_o

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

    Dk_mig_elyte = tk_elyte * sigma / params['DivCharge'] / ct.faraday