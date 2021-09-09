"""
    metal_air_single_particle.py

    Class file for metal air electrode methods
"""

import cantera as ct
import numpy as np

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

        # Electrode thickness and inverse thickness:
        self.dy = inputs['thickness']
        self.dyInv = 1/self.dy
        self.N_y = inputs['n-points'] # change to n-points

        # Phase volume fractions
        self.eps_host = inputs['eps_host']
        self.eps_oxide_init =inputs['eps_oxide']
        self.eps_elyte_init = 1 - self.eps_host - self.eps_oxide_init

        # This calculation assumes spherical particles of a single radius, with 
        # no overlap.
        # Electrode-electrolyte interface area, per unit geometric area.
        self.r_host = inputs['r_host']
        self.th_oxide = inputs['th_oxide']
        self.V_host = 4./3. * np.pi * (self.r_host / 2)**3  # carbon or host volume [m3]
        self.A_host = 4. * np.pi * (self.r_host / 2)**2    # carbon or host surface area [m2]
        self.A_init = self.eps_host * self.A_host / self.V_host  # m2 of interface / m3 of total volume [m-1]
        self.A_oxide = np.pi* inputs['d_oxide']**2/4.   # oxide area
        self.V_oxide = 2./3. * np.pi* (inputs['d_oxide']/2.)**2 * self.th_oxide #oxide volume

        # For some models, the elyte thickness is different from that of the 
        # electrode, so we specify is separately:
        self.dy_elyte = self.dy
        self.dy_elyte_node = self.dy_elyte/self.N_y

        # Inverse double layer capacitance, per unit interfacial area.
        self.C_dl_Inv = 1/inputs['C_dl']

        # Microstructure-based transport scaling factor, based on Bruggeman 
        # coefficient of -0.5:
        self.elyte_microstructure = self.eps_elyte_init**1.5 # where would we use this?
        
        # SV_offset specifies the index of the first SV variable for the 
        # electode (zero for anode, n_vars_anode + n_vars_sep for the cathode)
        self.SV_offset = offset

        # Determine the electrode capacity (Ah/m2)

        # Max voume concentration of the oxide (all elyte has been replaced by oxide)
        
        self.capacity = (inputs['stored-species']['charge']*ct.faraday
                 *self.eps_elyte_init)*inputs['thickness']/(3600
                 *inputs['stored-species']['MW']) 

        # Number of state variables: electrode potential, electrolyte composition, oxide volume fraction 
        self.n_vars = 3 + self.elyte_obj.n_species
        self.n_vars_tot = self.N_y*self.n_vars

        # This model produces zero plots, but might someday.
        self.n_plots = 0

        # Set Cantera object state:
        self.host_obj.TP = params['T'], params['P']
        self.elyte_obj.TP = params['T'], params['P']
        self.surf_obj.TP = params['T'], params['P']
        #self.conductor_obj.TP = params['T'], params['P']

        # Set up pointers to specific variables in the solution vector:
        self.SVptr = {}
        self.SVptr['phi_ed'] = np.arange(0, self.n_vars_tot, self.n_vars)
        self.SVptr['phi_dl'] = np.arange(1, self.n_vars_tot, self.n_vars)
        self.SVptr['eps_oxide'] = np.arange(2, self.n_vars_tot, self.n_vars)
        self.SVptr['C_k_elyte'] = np.ndarray(shape=(self.N_y, 
            self.elyte_obj.n_species), dtype='int')       
        for i in range(self.N_y):
            self.SVptr['C_k_elyte'][i,:] = range(3 + i*self.n_vars, 
                3 + i*self.n_vars + self.elyte_obj.n_species)

        # A pointer to where the SV variables for this electrode are, within the 
        # overall solution vector for the entire problem:
        self.SVptr['electrode'] = np.arange(offset, offset+self.n_vars_tot)

        # Save the indices of any algebraic variables:
        self.algvars = offset + self.SVptr['phi_ed'][:]

    def initialize(self, inputs, sep_inputs):

        # Initialize the solution vector for the electrode domain:
        SV = np.zeros(self.n_vars_tot)

        # Load intial state variables: Change it later
        SV[self.SVptr['phi_ed']] = inputs['phi_0']
        SV[self.SVptr['phi_dl']] = sep_inputs['phi_0'] - inputs['phi_0']
        SV[self.SVptr['eps_oxide']] = self.eps_oxide_init
        SV[self.SVptr['C_k_elyte']] = self.elyte_obj.concentrations
        
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
        resid = np.zeros((self.n_vars_tot,))

        # Save local copies of the solution vectors, pointers for this electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]
        SVdot_loc = SVdot[SVptr['electrode']]
        #Creation of temporary 
        dSVdt = np.zeros_like(SVdot_loc) # maybe change this
        i_io = np.zeros(self.N_y + 1) #
        i_el = np.zeros(self.N_y + 1)
        N_k_elyte = np.zeros_like(SV_loc[SVptr['C_k_elyte']]) # Change switch to N_K
        
        N_k_sep, i_io_sep = sep.electrode_boundary_flux(SV, self, params['T']) 
        j_index = np.arange(0,self.N_y+1, 1)

        # Look at node adjacent to sep interface 
        j = 0 #j = j_index[0]

        i_io[j] = i_io_sep   
        N_k_elyte[j,:] = N_k_sep # question: is this right?

        if self.name=='anode':
            # The electric potential of the anode = 0 V.
            resid[[SVptr['phi_ed'][0]]] = SV_loc[SVptr['phi_ed'][0]]
            # j_index = j_index[::-1]
        elif self.name=='cathode':
            # For the cathode, the potential of the cathode must be such that 
            # the electrolyte electric potential (calculated as phi_ca + 
            # dphi_dl) produces the correct ionic current between the separator # and cathode:
            if params['boundary'] == 'current':
                resid[SVptr['phi_ed'][0]] = i_io[0] - params['i_ext']
            elif params['boundary'] == 'potential':                  
                resid[SVptr['phi_ed']] = (SV_loc[SVptr['phi_ed']] 
                    - params['potential']) 


        phi_elyte, eps_oxide, eps_elyte, N_k_elyte[j+1,:], i_io[j+1] = \
            self.mass_fluxes(SV_loc, SVptr, params, j)
        self.elyte_obj.electric_potential = phi_elyte
        self.host_obj.electric_potential = SV[SVptr['phi_ed'][0]]

        ck_elyte = SV_loc[SVptr['C_k_elyte'][j]]
        self.elyte_obj.X = ck_elyte
        #change in volume fraction
        A_avail = self.A_init - eps_oxide/self.th_oxide
        A_surf_ratio = A_avail*self.dy
        sdot_product = self.surf_obj.get_net_production_rates(self.product_obj)
        dSVdt[SVptr['eps_oxide'][j]] =  A_avail * np.dot(sdot_product, 
            self.product_obj.partial_molar_volumes)  

        #Change in potential
        sdot_electron = self.surf_obj.get_net_production_rates(self.host_obj)
        i_Far = -(ct.faraday * sdot_electron)
        i_dl = self.i_ext_flag*(i_io[j] - i_io[j+1])/A_surf_ratio - i_Far
        dSVdt[SVptr['phi_dl'][j]] = - i_dl*self.C_dl_Inv
        #change in concentration
        sdot_elyte_c = self.surf_obj.get_net_production_rates(self.elyte_obj) 
        sdot_elyte_c[self.index_Li] -= i_dl / ct.faraday 
        
        dSVdt[SVptr['C_k_elyte'][j]] = ((N_k_elyte[j] - N_k_elyte[j+1]) + 
            sdot_elyte_c * A_surf_ratio) * self.dyInv
        
        for j in j_index[1:-2]:
            resid[[SVptr['phi_ed'][j]]] = (SV_loc[[SVptr['phi_ed'][j]]] 
                - SV_loc[[SVptr['phi_ed'][0]]])
            print(j, resid[[SVptr['phi_ed'][j]]], SV_loc[[SVptr['phi_ed'][j]]] )
            phi_elyte, eps_oxide, eps_elyte, N_k_elyte[j+1,:], i_io[j+1] = self.mass_fluxes(SV_loc, SVptr, params, j)
            #volume fractions
            self.elyte_obj.electric_potential = phi_elyte
            self.host_obj.electric_potential = SV[SVptr['phi_ed'][j]]

            ck_elyte = SV_loc[SVptr['C_k_elyte'][j]]
            self.elyte_obj.X = ck_elyte
            
            self.elyte_microstructure = eps_elyte**1.5
                                        
            #Calculation of change in volume fraction
            A_avail = self.A_init - eps_oxide/self.th_oxide
            A_surf_ratio = A_avail*self.dy
            sdot_cath = self.surf_obj.get_net_production_rates(self.product_obj)
            dSVdt[SVptr['eps_oxide'][j]] =  A_avail * np.dot(sdot_cath, self.product_obj.partial_molar_volumes)  

            #caluculate double layer 
            sdot_electron = self.surf_obj.get_net_production_rates(self.host_obj)
            i_Far = -(ct.faraday * sdot_electron)
            i_io[j] =  params['i_ext']
            i_dl = self.i_ext_flag*(i_io[j] - i_io[j+1])/A_surf_ratio - i_Far
            dSVdt[SVptr['phi_dl'][j]] = - i_dl*self.C_dl_Inv

            #calculate change in concentration
            sdot_elyte_c = self.surf_obj.get_net_production_rates(self.elyte_obj) 
            sdot_elyte_c[self.index_Li] -= i_dl / ct.faraday 
            
            dSVdt[SVptr['C_k_elyte'][j]] = (-N_k_elyte[j] + N_k_elyte[j+1]) + (sdot_elyte_c * A_surf_ratio ) * self.dyInv / eps_elyte

                       
        j = self.N_y-1 
        resid[[SVptr['phi_ed'][j]]] = (SV_loc[[SVptr['phi_ed'][j]]] 
                - SV_loc[[SVptr['phi_ed'][0]]])

        phi_ed = SV[SVptr['phi_ed'][j]]
        phi_elyte = phi_ed + SV[SVptr['phi_dl'][j]]
        eps_oxide = SV_loc[SVptr['eps_oxide'][j]]
        eps_elyte = 1 - eps_oxide - self.eps_host
        #volume fractions
        self.elyte_obj.electric_potential = phi_elyte
        self.host_obj.electric_potential = phi_ed

        ck_elyte = SV_loc[SVptr['C_k_elyte'][j]]
        self.elyte_obj.X = ck_elyte
        
        self.elyte_microstructure = eps_elyte**1.5
        #Calculation of change in volume fraction
        A_avail = self.A_init - eps_oxide/self.th_oxide
        A_surf_ratio = A_avail*self.dy
        sdot_cath = self.surf_obj.get_net_production_rates(self.product_obj)
        dSVdt[SVptr['eps_oxide'][j]] =  A_avail * np.dot(sdot_cath, self.product_obj.partial_molar_volumes)  

        #caluculate double layer 
        sdot_electron = self.surf_obj.get_net_production_rates(self.host_obj)
        i_Far = -(ct.faraday * sdot_electron)
        i_io[j+1] =  params['i_ext']
        i_dl = self.i_ext_flag*(i_io[j] - i_io[j+1])/A_surf_ratio - i_Far
        dSVdt[SVptr['phi_dl'][j]] = - i_dl*self.C_dl_Inv

        #calculate change in concentration
        sdot_elyte_c = self.surf_obj.get_net_production_rates(self.elyte_obj) 
        sdot_elyte_c[self.index_Li] -= i_dl / ct.faraday 
        sdot_elyte_o =  self.air_elyte_obj.get_net_production_rates(self.elyte_obj)
        
        dSVdt[SVptr['C_k_elyte'][j]] = -N_k_elyte[j-1] + (sdot_elyte_c * A_surf_ratio + sdot_elyte_o*eps_elyte)

        # residual formulas
        resid[SVptr['phi_dl']] = (SVdot_loc[SVptr['phi_dl']] - dSVdt[SVptr['phi_dl']])
        resid[SVptr['C_k_elyte']] = (SVdot_loc[SVptr['C_k_elyte']] - dSVdt[SVptr['C_k_elyte']])
        resid[SVptr['eps_oxide']] = (SVdot_loc[SVptr['eps_oxide']] - dSVdt[SVptr['eps_oxide']])
            
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

    #What is the small here? Is it to avoid zeros?
    #def read_state():
    

    def mass_fluxes(self, SV_loc, SVptr, params, j):
        
        #FIX: Actual diffusion  Question: best way to pull this 
        D_k = {}
        D_k['Li+[elyt])'] = 4e-11          # bulk diff coeff Li+ in elyte (m2/s)
        D_k['TFSI-[elyt]'] = 4e-13         # bulk diff coeff PF6- in elyte (m2/s)
        D_k['O2(e)'] = 7e-12           # bulk diff coeff O2 in elyte (m2/s)
        D_k['C10H22O5[elyt]'] = 1.           # EC diffusion is fast
        D_k['Li2O2[elyt]'] = 1.           # EC diffusion is fast

        D_k_temp = np.array([1.11e-10, 6.98e-11, 8.79e-11, 4.26e-11, 2e-13])
        phi_ed = SV_loc[SVptr['phi_ed'][j]]
        phi_elyte = phi_ed + SV_loc[SVptr['phi_dl'][j]]

        phi_ed_next = SV_loc[SVptr['phi_ed'][j+1]]
        phi_elyte_next = phi_ed_next + SV_loc[SVptr['phi_dl'][j+1]]
        #convert from mol/m3 to kg/m3
        rho_k = SV_loc[SVptr['C_k_elyte'][j]] 
        rho_k_next = SV_loc[SVptr['C_k_elyte'][j+1]]
       
        eps_oxide = SV_loc[SVptr['eps_oxide'][j]]
        eps_elyte = 1 - eps_oxide - self.eps_host
        eps_oxide_next =  SV_loc[SVptr['eps_oxide'][j+1]]
        eps_elyte_next = 1 - eps_oxide_next - self.eps_host
       # Take averages to find interface values.  Eventually this should be 
       # weighted by the volume dimensions:
        rho_k_avg = (rho_k + rho_k_next)/2.
        eps_avg = (eps_elyte + eps_elyte_next)/2.

        D_k_elyte = D_k_temp * eps_avg**1.5
        D_k_mig = D_k_elyte*self.elyte_obj.charges*ct.faraday/(ct.gas_constant*params['T'])*rho_k_avg#Question: easiest way to access this from yaml file
        #Question: is that right for cantera charages?

        #Question: is this a rate? or is this a concentration?  Re: Amy's code
        
        N_k = (D_k_elyte*(rho_k/eps_elyte- rho_k_next/eps_elyte_next) + D_k_mig*(phi_elyte - phi_elyte_next))*self.dyInv

        i_io = np.dot(N_k, self.elyte_obj.charges)*ct.faraday

        return  phi_elyte, eps_oxide, eps_elyte, N_k, i_io

#Official Soundtrack:
    #Jimmy Eat World - Chase the Light + Invented
    #Lay - Lit
    #George Ezra - Staying at Tamara's