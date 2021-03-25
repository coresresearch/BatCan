def residual(SV, SVdot, self, sep, counter, params):
    import numpy as np
    import cantera as ct
    from math import tanh
    
    resid = np.zeros((self.nVars,))

    # Save local copies of the solution vectors, pointers for this electrode:
    SVptr = self.SVptr
    SV_loc = SV[SVptr['residual']]
    SVdot_loc = SVdot[SVptr['residual']]

    # Read and set electrolyte electric potential:
    phi_ed = SV_loc[SVptr['phi_ed']]
    phi_elyte = phi_ed + SV_loc[SVptr['phi_dl']]

    # Set electric potentials for Cantera objects:
    self.bulk_obj.electric_potential = phi_ed
    self.conductor_obj.electric_potential = phi_ed
    self.elyte_obj.electric_potential = phi_elyte
    
    # Faradaic current density is positive when electrons are consumed 
    # (Li transferred to the anode)
    sdot_electron = self.surf_obj.get_net_production_rates(self.conductor_obj)
    i_Far = -ct.faraday*sdot_electron
    
    # Double layer current has the same sign as i_Far:
    i_dl = self.i_ext_flag*params['i_ext']/self.A_surf_ratio - i_Far
    
    if self.name=='anode':
        resid[SVptr['residual'][SVptr['phi_ed']]] = SV_loc[SVptr['phi_ed']]
    elif self.name=='cathode':
        # TEMPORARY: phi_elyte in cathode matches that in the anode.
        # TODO #21
        N_k_sep = sep.cathode_boundary(SV, self, sep)
        i_io = np.dot(N_k_sep, self.elyte_obj.charges)*ct.faraday
                
        resid[SVptr['phi_ed']] = i_io - params['i_ext']

    resid[SVptr['phi_dl']] = (SVdot_loc[SVptr['phi_dl']] - i_dl*self.C_dl_Inv)
    
    # Production rate of electrode species (kmol/m2/s)
    mult = tanh(SV_loc[SVptr['thickness']]/self.min_thickness)
    sdot_electrode = (self.surf_obj.get_creation_rates(self.bulk_obj)
            - mult*self.surf_obj.get_destruction_rates(self.bulk_obj))
    
    # Change in thickness per time:
    dH_dt = np.dot(sdot_electrode,self.bulk_obj.partial_molar_volumes)
    resid[SVptr['thickness']] = SVdot_loc[SVptr['thickness']] - dH_dt
    
    # Set time derivatives to zero (temporary)
    resid[SVptr['C_k_elyte']] = SVdot_loc[SVptr['C_k_elyte']]

    return resid

def make_alg_consistent(SV, an, sep, ca, params):
    return SV
