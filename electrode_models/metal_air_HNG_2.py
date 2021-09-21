"""
    metal_air_single_particle.py
    Class file for metal air electrode methods
"""

import cantera as ct
import numpy as np
import math

class electrode(): 
    """
    Create an electrode object representing the metal air electrode.
    """

    def __init__(self, input_file, inputs, sep_inputs, counter_inputs,    
        electrode_name, params, offset):
        """
        Initialize the model.
        """ 
   
        # Import relevant Cantera objects.
        self.gas_obj = ct.Solution(input_file, inputs['gas-phase'])
        self.elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])
        self.air_elyte_obj = ct.Interface(input_file, inputs['elyte-iphase'], [self.gas_obj, self.elyte_obj])
        self.host_obj = ct.Solution(input_file, inputs['host-phase'])
        self.product_obj = ct.Solution(input_file, inputs['product-phase'])
        self.surf_obj = ct.Interface(input_file, inputs['surf-iphase'], 
            [self.product_obj, self.elyte_obj, self.host_obj])

        # Anode or cathode? Positive external current delivers positive charge 
        # to the anode, and removes positive charge from the cathode.
        self.name = electrode_name
        if self.name=='anode':
            self.i_ext_flag = -1
        elif self.name=='cathode':
            self.i_ext_flag = 1
        else:
            raise ValueError("Electrode must be an anode or a cathode.")

        # Store the species index of the Li ion in the Cantera object for the 
        # electrolyte phase:
        self.index_Li = self.elyte_obj.species_index(inputs['mobile-ion'])
        self.index_LiO2 = self.elyte_obj.species_index(inputs['reactive-ion'])

        # Electrode thickness and inverse thickness:
        self.dy = inputs['thickness']
        self.dyInv = 1/self.dy

        #number of bin
        self.n_bins = inputs['buckets']
        # Phase volume fractions
        self.eps_host = inputs['eps_host']
        self.eps_oxide_int =inputs['eps_oxide']
        self.eps_elyte_int = 1 - self.eps_host - self.eps_oxide_int

        # # This calculation assumes spherical particles of a single radius, with 
        # # no overlap.
        # # Electrode-electrolyte interface area, per unit geometric area.
        self.r_host = inputs['r_host']
        # self.th_oxide = inputs['th_oxide']
        self.V_host = 4./3. * np.pi * (self.r_host / 2.)**3  # carbon or host volume [m3]
        self.A_host = 4. * np.pi * (self.r_host / 2.)**2    # carbon or host surface area [m2]
        self.A_init = self.eps_host * self.A_host / self.V_host  # m2 of interface / m3 of total volume [m-1]
        
        # inputs for nucleation:
        self.c_li_sat = inputs['C_Li_Sat']
        self.c_liO2_sat = inputs['C_LiO2_Sat']
        self.gamma_surf = inputs['Gamma_surf']
        self.k_surf = inputs['k_surf']
        self.k_surf_des = inputs['k_surf_des']
        self.phi = (2. + math.cos(inputs['contact-angle']))*(1.- math.cos( inputs['contact-angle']))**2./4.
        self.d_li = inputs['DLi']
        self.radius = np.linspace(inputs['Minrad'], inputs['Maxrad'], self.n_bins + 1)
        self.bin_width = self.radius[2] - self.radius[1]
        
        # For some models, the elyte thickness is different from that of the 
        # electrode, so we specify is separately:
        self.dy_elyte = self.dy


        # Inverse double layer capacitance, per unit interfacial area.
        self.C_dl_Inv = 1/inputs['C_dl']

        # Microstructure-based transport scaling factor, based on Bruggeman 
        # coefficient of -0.5:
        self.elyte_microstructure = self.eps_elyte_int**1.5 # where would we use this?
        
        # SV_offset specifies the index of the first SV variable for the 
        # electode (zero for anode, n_vars_anode + n_vars_sep for the cathode)
        self.SV_offset = offset

        # Determine the electrode capacity (Ah/m2)

        # Max voume concentration of the oxide (all elyte has been replaced by oxide)
        
        self.capacity = (inputs['stored-species']['charge']*ct.faraday
                 *self.eps_elyte_int)*inputs['thickness']/(3600
                 *self.product_obj[inputs['stored-species']['name']].partial_molar_volumes[0]) 

        # Number of state variables: electrode potential, electrolyte composition, oxide volume fraction 
        self.n_vars = 2 + self.elyte_obj.n_species + self.n_bins + 1

        # This model produces zero plots, but might someday.
        self.n_plots = 0

        # Set Cantera object state:
        self.host_obj.TP = params['T'], params['P']
        self.elyte_obj.TP = params['T'], params['P']
        self.surf_obj.TP = params['T'], params['P']
        #self.conductor_obj.TP = params['T'], params['P']

        # Set up pointers to specific variables in the solution vector:
        self.SVptr = {}
        self.SVptr['phi_ed'] = np.array([0])
        self.SVptr['phi_dl'] = np.array([1])
        self.SVptr['C_k_elyte'] = np.arange(2, 2 + self.elyte_obj.n_species)
        self.SVptr['histogram'] = np.arange(2 + self.elyte_obj.n_species, 2 + self.elyte_obj.n_species + self.n_bins +1)
        
        # There is only one node, but give the pointer a shape so that SVptr
        # ['C_k_elyte'][j] accesses the pointer array:
        self.SVptr['C_k_elyte'].shape = (1,self.elyte_obj.n_species)

        # A pointer to where the SV variables for this electrode are, within the 
        # overall solution vector for the entire problem:
        self.SVptr['electrode'] = np.arange(offset, offset+self.n_vars)

        # Save the indices of any algebraic variables:
        self.algvars = offset + self.SVptr['phi_ed'][:]

    def initialize(self, inputs, sep_inputs):

        # Initialize the solution vector for the electrode domain:
        SV = np.zeros([self.n_vars])

        # Load intial state variables:
        SV[self.SVptr['phi_ed']] = inputs['phi_0']
        SV[self.SVptr['phi_dl']] = sep_inputs['phi_0'] - inputs['phi_0']
        SV[self.SVptr['C_k_elyte']] = self.elyte_obj.concentrations
        SV[self.SVptr['histogram']] = np.zeros_like(self.radius)
        
        return SV

    def residual(self, t, SV, SVdot, sep, counter, params):
        """
        Define the residual for the state of the metal air electrode.
        This is an array of differential and algebraic governing equations, one for each state variable in the anode (anode plus a thin layer of electrolyte + separator).
        1. The electric potential in the electrode phase is an algebraic variable.
            In the anode, phi = 0 is the reference potential for the system.
            In the cathode, the electric potential must be such that the ionic current is spatially invariant (i.e. it is constant and equal to the external applied current, for galvanostatic simulations).  
            The residual corresponding to these variables (suppose an index 'j') are of the form:
                resid[j]  = (epression equaling zero)
        2. All other variables are governed by differential equations.
        
            We have a means to calculate dSV[j]/dt for a state variable SV[j] (state variable with index j).  
        
            The residuals corresponding to these variables will have the form:
                resid[j] = SVdot[j] - (expression equalling dSV/dt)
        Inputs:
            - SV: the solution vector representing the state of the entire battery domain.
            - SVdot: the time derivative of each state variable: dSV/dt
            - electrode: the object representing the current electrode
            - sep: the object representing the separator
            - counter: the object representing the electrode counter to the current electrode
            - params: dict of battery simulation parameters.
        """
        
        # Initialize the residual:
        resid = np.zeros((self.n_vars,))

        # Save local copies of the solution vectors, pointers for this electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]
        SVdot_loc = SVdot[SVptr['electrode']]

        # Read the electrode and electrolyte electric potential:
        phi_ed = SV_loc[SVptr['phi_ed']]
        phi_elyte = phi_ed + SV_loc[SVptr['phi_dl']]      
        
        Dr_dt = np.zeros_like(SV[SVptr['histogram']])
        dhistogram_dt = np.zeros_like(SV[SVptr['histogram']])
        Histogram = SV_loc[SVptr['histogram']] #nucleations m-2

        eps_oxide = self.eps_oxide_int + np.sum(Histogram*(self.radius**3*math.pi*2./3.))*self.dyInv
        eps_elyte = 1 - eps_oxide - self.eps_host
        # Set electric potentials for Cantera objects:
        self.host_obj.electric_potential = phi_ed
        #self.conductor_obj.electric_potential = phi_ed
        self.elyte_obj.electric_potential = phi_elyte

        self.elyte_microstructure = eps_elyte**1.5

        ck_elyte = SV_loc[SVptr['C_k_elyte']]

        self.elyte_obj.X = ck_elyte

        ck_elyte = SV_loc[SVptr['C_k_elyte'][0]]
        
        sdot_electron = self.surf_obj.get_net_production_rates(self.host_obj) #kmol m-2 s-2
        
        # Faradaic current density is positive when electrons are consumed 
        # (Li transferred to the electrode)
        i_Far = -(ct.faraday * sdot_electron) #[A m-2 of host electrolyte interface]
        
        # Calculate the electrolyte species fluxes and associated ionic current 
        # at the boundary with the separator:
        N_k_sep, i_io = sep.electrode_boundary_flux(SV, self, params['T']) #?
        
        if self.name=='anode':
            # The electric potential of the anode = 0 V.
            resid[[SVptr['phi_ed'][0]]] = SV_loc[SVptr['phi_ed'][0]]
            
        elif self.name=='cathode':
            # For the cathode, the potential of the cathode must be such that 
            # the electrolyte electric potential (calculated as phi_ca + 
            # dphi_dl) produces the correct ionic current between the separator # and cathode:
            if params['boundary'] == 'current':
                resid[SVptr['phi_ed']] = i_io - params['i_ext']
            elif params['boundary'] == 'potential':                  
                resid[SVptr['phi_ed']] = (SV_loc[SVptr['phi_ed']] 
                    - params['potential']) 
        # Molar production rate of electrode species (kmol/m2/s). Should be seperate on the discretization.
        sdot_elyte_o = self.air_elyte_obj.get_net_production_rates(self.elyte_obj)
        sdot_elyte_c = self.surf_obj.get_net_production_rates(self.elyte_obj) 
        
        
        # Double layer current has the same sign as i_Far, and is based on 
        # charge balance in the electrolyte phase:
        A_avail = self.A_init #- np.sum(Histogram*self.radius)  #m2 interface/ m3 total volume [m-1]
        A_surf_ratio = A_avail*self.dy # m2 interface / m2 total area [-]
        i_dl = self.i_ext_flag*i_io/A_surf_ratio - i_Far #does this need to be changed? #units of i_io?? A m-2 surface area
        
        #preliminary 
        V = 1.98E-5  # 1.98E-5 #m3 mol-1 // molar volume// Yin (2017) so we just get the array from self.product_phase.molar_volume
        a_d = (ck_elyte[self.index_LiO2]*ct.avogadro)**(-1./3.) # length scale of diffusion
        r_crit = 1E-10 # 2.*self.gamma_surf*V/(ct.gas_constant*params['T']*math.log(ck_elyte[self.index_LiO2]/self.c_liO2_sat*ck_elyte[self.index_Li]/self.c_li_sat))  # m // critical radius
        #there's something about using cantera here for molar volume but I'm not sure
        N_crit = 4./3.*math.pi*r_crit**3.*ct.avogadro/V # number of molecules in the critical nucleus of size
        Del_G_Crit = self.phi*4./3.*math.pi*self.gamma_surf*r_crit**2. # J mol-1 // energy barrier of the nucleation
        
        if N_crit <0:
            Del_G_Crit =0
        Z = math.sqrt(Del_G_Crit/(self.phi*3*math.pi*ct.boltzmann*params['T']*N_crit)) # - // Zeldovich factor #forgot how to fix, DeCaluwe should commit his code
        V_crit = 4./3.*math.pi*r_crit**3. # m3 // Critical volume
        N_sites = A_surf_ratio/(math.pi*r_crit**2) # number of nucleation sites #m-2 [total area]
         
        k_nuc= self.d_li*(a_d**-2)  #nucleations/s

        DN_Dt = 1E-13*k_nuc*N_sites*Z*math.exp(-Del_G_Crit/(ct.boltzmann*params['T'])) #nuc/m2
        print(DN_Dt)
        #DN_Dt = DN_Dt*math.exp(2.*0.5*8.854E-12*2.91/(ct.boltzmann*params['T']))*math.exp(0.5*8.854E-12*phi_elyte/(ct.boltzmann*params['T']))
        #calculate for loop to get Histogram
        for i, r in enumerate(self.radius):
                if r > r_crit:
                    dhistogram_dt[i] += DN_Dt
                    break    
        for i, N in enumerate(Histogram):
            Dr_dt[i] =  self.d_li*V*(ck_elyte[self.index_LiO2]- self.c_li_sat)*(ck_elyte[self.index_LiO2]
                -self.c_liO2_sat)/(self.radius[i]+self.d_li/self.k_surf)- math.pi*self.radius[i]**2*N*self.gamma_surf*self.k_surf_des
            dNdt_radii = Dr_dt[i]/self.bin_width*N
            if dNdt_radii <0:
                dhistogram_dt[i] += dNdt_radii
                if i > 0:
                    dhistogram_dt[i-1] -= dNdt_radii
            elif dNdt_radii > 0 and self.radius[i] != self.radius[-1]:
                dhistogram_dt[i] -= dNdt_radii
                dhistogram_dt[i+1] += dNdt_radii
       
        # Differential equation for the double layer potential:
        resid[SVptr['phi_dl']] = \
            SVdot_loc[SVptr['phi_dl']] - i_dl*self.C_dl_Inv

        
        # Double layer current removes Li from the electrolyte.  Subtract this 
        # from sdot_electrolyte: kmol m-2 s-2
        sdot_elyte_c[self.index_Li] -= i_dl / ct.faraday - DN_Dt*V_crit/V - 2.*np.sum(dhistogram_dt*Dr_dt*self.radius
            *self.radius)*np.pi/V
        sdot_elyte_c[self.index_LiO2] -= i_dl / ct.faraday - DN_Dt*V_crit/V - 2.*np.sum(dhistogram_dt*Dr_dt*self.radius
            *self.radius)*np.pi/V
            
<<<<<<< HEAD
        # Change in electrolyte species concentration per unit time (mol m-3 s-1):
=======
        # Change in electrolyte species concentration per unit time (kmol m-2 electrolyte area s-1):
>>>>>>> 903853f (Semi-functional code (runs undersome conditions))
        dCk_elyte_dt = \
            ((sdot_elyte_c + sdot_elyte_o + self.i_ext_flag * N_k_sep) 
            * self.dyInv / eps_elyte) # first term is reaction second term is seperater? 
        resid[SVptr['C_k_elyte']] = SVdot_loc[SVptr['C_k_elyte']] - dCk_elyte_dt
        #molar production rate of 
        #sdot_cath = self.surf_obj.get_net_production_rates(self.product_obj)
        # available interface area on carbon particle
        resid[SVptr['histogram']] = (SVdot_loc[SVptr['histogram']] - dhistogram_dt)
        return resid
        
    def voltage_lim(self, SV, val):
        """
        Check to see if the voltage limits have been exceeded.
        """
        # Save local copies of the solution vector and pointers for this electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]
        
        # Calculate the current voltage, relative to the limit.  The simulation 
        # looks for instances where this value changes sign (i.e. crosses zero)    
        voltage_eval = SV_loc[SVptr['phi_ed']] - val
        
        return voltage_eval

    def adjust_separator(self, sep):
        """ 
        Sometimes, an electrode object requires adjustments to the separator object.  This is not the case, for the SPM.
        """

        # Return the separator class object, unaltered:
        return sep

    def output(self, axs, solution, ax_offset):
        """Plot the intercalation fraction vs. time"""
        C_k_an = solution[3 + self.SV_offset]
        axs[ax_offset].plot(solution[0,:]/3600, C_k_an)
        axs[ax_offset].set_ylabel(self.name+' Li \n(kmol/m$^3$)')
        axs[ax_offset].set(xlabel='Time (h)')

        return axs

#Official Soundtrack:
    #Jimmy Eat World - Chase the Light + Invented
    #Leslie Cheung - 這些年來(台灣發行)
    #Various artists - The Metallica Blacklist