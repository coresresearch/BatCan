"""
Author:
Amy LeBar 24 April 2019

Diffusion coefficient function
Inputs: bulk diffusion coefficents,

Outputs: chemical diffusion coefficient (D_k_elyte), and migration term (D_k_mig_elyte)
"""
# Load needed modules
import cantera as ct
import numpy as np

class dst:
    
    Dk_el_0 = []
    C_k = []
    T_elyte = []
    ptr_el = []
    params = []
    
    def coeffs():
        Dk_el = dst.Dk_el_0
        Dk_migr_el = dst.Dk_el_0*dst.C_k
        
        return Dk_el, Dk_migr_el
    
    "========================================================================="
    
class cst:
    
    Dk_el_0 = []
    C_k = []
    ptr_el = []
    z_k = []
    rho_bar = []
    params = []
    T_0 = []
    T = []

    def coeffs():
        z_k = cst.z_k; C_k = cst.C_k; ptr_el = cst.ptr_el; params = cst.params
        T_0 = cst.T_0; T = cst.T
        C_Li = C_k[cst.ptr_el['Li']]            # Concentration of lithium
        C_sol = C_k[cst.ptr_el['solvents']]     # Concentration of the solvent
        
        Dk_el = params['Dk_elyte_o']
        
        # Polynomial fit for chemical diffusion coefficient of lithium in elyte       
        Dk_el[cst.ptr_el['Li']] = ((params['D_Li_a']*C_Li**2 
                          + params['D_Li_b']*C_Li + params['D_Li_c'])
                          *(params['D_Li_d']*C_Li**2 + params['D_Li_e']*C_Li 
                          + params['D_Li_f'])/(params['D_Li_g'] 
                          + params['D_Li_h']*C_Li + params['D_Li_i']*C_Li**2 
                          + params['D_Li_j']*C_Li**3)*sum(cst.C_k)/sum(C_sol))
    
        # Modify diffusion coefficient term to include an Arrhenius-like
        #   temperature dependence
        Dk_el[cst.ptr_el['Li']] = Dk_el[cst.ptr_el['Li']]*np.exp((T-T_0)/ct.gas_constant/T_0)
        
        # Assume diffusion coefficient for cation same as anion
        Dk_el[cst.ptr_el['PF6']] = Dk_el[cst.ptr_el['Li']]
        
        # Use a polynomial fit to calculate concentration dependence of the
        #   ionic transference number
        tk_el_0 = (params['t_elyte_a'] + params['t_elyte_b']*C_Li
                 + params['t_elyte_c']*C_Li**2 + params['t_elyte_d']*C_Li**3)
        
        # Set tk for non-charged species to zero and set t_PF6 to 1 - t_Li
        tk_el = abs(z_k)*((z_k + 1)*tk_el_0/2. - (z_k - 1)*(1 - tk_el_0)/2.)
        
        # Effective conductivity comes out as S/dm, need to multiply by 10 
        sigma = (params['sigma_elyte_a']*C_Li 
               + params['sigma_elyte_b']*C_Li**1.5
               + params['sigma_elyte_c']*C_Li**3)
        
        sigma = sigma*10*np.exp((T-T_0)/ct.gas_constant/T_0)
        
        # Polynomial fit for thermodynamic factor: (1 + d(ln f)/d(ln C_Li))
        thermo_factor = ((params['gamma_elyte_a']*C_Li**2 
                       + params['gamma_elyte_b']*C_Li + params['gamma_elyte_c'])
                       /(params['gamma_elyte_d'] + params['gamma_elyte_e']*C_Li
                       + params['gamma_elyte_f']*C_Li**2 
                       + params['gamma_elyte_g']*C_Li**3))
                       
        # Diffusional conductivity
        C = cst.rho_bar
        sigma_D = 2*(tk_el[ptr_el['Li']] - 1)*C*thermo_factor/z_k[ptr_el['Li']] \
                / ct.faraday

        Dk_el = Dk_el + sigma_D*tk_el/C_k/params['divCharge']/ct.faraday
        
        Dk_migr_el = tk_el*sigma/params['divCharge']/ct.faraday
        
        return Dk_el, Dk_migr_el
    
class cst_2:
    
    Dk_el_0 = []
    C_k = []
    ptr_el = []
    z_k = []
        
#        ptr_el = cst.ptr_el
#        C_k = cst.C_k
#        q_Li = [key.startswith('Li') for key in ptr_el]
#        Li_key = [list(ptr_el.keys())[i] for i in range(len(q_Li)) if q_Li[i]]
#        Li_ind = ptr_el[Li_key]
#        C_Li = C_k[Li_ind]
        
        
    
"=============================== Old functions ==============================="

## Dilute Solution Theory (dst)
#def dst_fun(Ck_elyte,params):
#    Dk_elyte = params['Dk_elyte_o']
#    Dk_mig_elyte = params['Dk_mig_elyte_o'] * Ck_elyte
#
#    return Dk_elyte, Dk_mig_elyte
#
## Concentrated Solution Theory (cst)
#def cst_fun(Ck_elyte,objs,params):
#    elyte = objs['elyte']
#
#    C_Li = Ck_elyte[params['iLi_elyte']]*1000
#    C_EC = Ck_elyte[params['iEC_elyte']]*1000
#    C_EMC = Ck_elyte[params['iEMC_elyte']]*1000
#
#    Dk_elyte = params['Dk_elyte_o']
#
#    # Chemical diffusion coefficent for Li+ and PF6- in elyte
#    Dk_elyte[params['iLi_elyte']] = (params['D_Li_a']*C_Li**2 + \
#                                params['D_Li_b']*C_Li + params['D_Li_c']) * \
#                                (params['D_Li_d']*C_Li**2 + params['D_Li_e']*C_Li + \
#                                params['D_Li_f']) / (params['D_Li_g'] + params['D_Li_h']*C_Li + \
#                                params['D_Li_i']*C_Li**2 + params['D_Li_j']*C_Li**3) * \
#                                sum(1000*Ck_elyte) / (C_EC + C_EMC)
#
#    Dk_elyte[params['iPF6_elyte']] = Dk_elyte[params['iLi_elyte']]
#
#    # Positive ion transference number
#    tk_elyte_o = params['t_elyte_a'] + params['t_elyte_b']*C_Li + \
#                  params['t_elyte_c']*C_Li**2 + params['t_elyte_d']*C_Li**3
#
#    tk_elyte = abs(params['polarity_k']) * \
#            ((params['polarity_k'] + 1) * tk_elyte_o/2 - \
#            (params['polarity_k'] - 1) * (1 - tk_elyte_o)/2)
#    
#    # Effective conductivity
#    sigma = params['sigma_elyte_a']*C_Li + \
#            params['sigma_elyte_b']*C_Li**1.5 + \
#            params['sigma_elyte_c']*C_Li**3
#
#    # Thermodynamic factor - (1 - df/d(ln C_Li))
#    thermo_factor = (params['gamma_elyte_a']*C_Li**2 + \
#            params['gamma_elyte_b']*C_Li + params['gamma_elyte_c']) / \
#            (params['gamma_elyte_d'] + params['gamma_elyte_e']*C_Li + \
#            params['gamma_elyte_f']*C_Li**2 + params['gamma_elyte_g']*C_Li**3)
#
#    # Diffusional conductivity
#    sigma_D = 2*ct.gas_constant*elyte.TP[0] * \
#              sigma*(tk_elyte_o - 1)*thermo_factor / ct.faraday
#
#    Dk_elyte = Dk_elyte + sigma_D*tk_elyte / (Ck_elyte*params['divCharge']*ct.faraday)
#
#    Dk_mig_elyte = tk_elyte * sigma / params['divCharge'] / ct.faraday
#
#    return Dk_elyte, Dk_mig_elyte
