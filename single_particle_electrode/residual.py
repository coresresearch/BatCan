def residual(SV, self, params):
    import numpy as np
    import cantera as ct
    
    SV_dot = np.zeros((self.nVars,))
    
    # Read and set electrolyte electric potential:
    phi_ed = SV[self.SV_offset + self.SVptr['phi_ed']]
    phi_elyte = phi_ed + SV[self.SV_offset + self.SVptr['phi_dl']]
    self.bulk_obj.electric_potential = phi_ed
    self.conductor_obj.electric_potential = phi_ed
    self.elyte_obj.electric_potential = phi_elyte
    
    # Faradaic current density is positive when electrons are consumed 
    # (Li transferred to the anode)
    sdot_electron = self.surf_obj.get_net_production_rates(self.conductor_obj)
    i_Far = -ct.faraday*sdot_electron
    
    # Double layer current has the same sign as i_Far:
    i_dl = self.i_ext_flag*params['i_ext']/self.A_surf_ratio - i_Far
    
    SV_dot[self.SVptr['phi_dl']] = i_dl*self.C_dl_Inv

    return SV_dot