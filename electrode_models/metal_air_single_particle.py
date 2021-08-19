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

        # Phase volume fractions
        self.eps_host = inputs['eps_host']
        self.eps_oxide_int =inputs['eps_oxide']
        self.eps_elyte_int = 1 - self.eps_host - self.eps_oxide_int

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
                 *inputs['stored-species']['MW']) 

        # Number of state variables: electrode potential, electrolyte composition, oxide volume fraction 
        self.n_vars = 3 + self.elyte_obj.n_species

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
        self.SVptr['eps_oxide'] = np.array([2])
        self.SVptr['C_k_elyte'] = np.arange(3, 3 + self.elyte_obj.n_species)
        
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
        SV[self.SVptr['eps_oxide']] = self.eps_oxide_int
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
        resid = np.zeros((self.n_vars,))

        # Save local copies of the solution vectors, pointers for this electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]
        SVdot_loc = SVdot[SVptr['electrode']]

        # Read the electrode and electrolyte electric potential:
        phi_ed = SV_loc[SVptr['phi_ed']]
        phi_elyte = phi_ed + SV_loc[SVptr['phi_dl']]
        eps_oxide = SV_loc[SVptr['eps_oxide']]
        eps_elyte = 1 - eps_oxide - self.eps_host

        # Set electric potentials for Cantera objects:
        self.host_obj.electric_potential = phi_ed
        #self.conductor_obj.electric_potential = phi_ed
        self.elyte_obj.electric_potential = phi_elyte

        self.elyte_microstructure = eps_elyte**1.5

        ck_elyte = SV_loc[SVptr['C_k_elyte']]
        
        self.elyte_obj.X = ck_elyte

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

        # Double layer current has the same sign as i_Far, and is based on 
        # charge balance in the electrolyte phase:
        A_avail = self.A_init - eps_oxide/self.th_oxide  #m2 interface/ m3 total volume [m-1]
        A_surf_ratio = A_avail*self.dy # m2 interface / m2 total area [-]
        i_dl = self.i_ext_flag*i_io/A_surf_ratio - i_Far #does this need to be changed? #units of i_io?? A m-2 surface area

        # Differential equation for the double layer potential:
        resid[SVptr['phi_dl']] = \
            SVdot_loc[SVptr['phi_dl']] - i_dl*self.C_dl_Inv

        # Molar production rate of electrode species (kmol/m2/s). Should be seperate on the discretization.
        sdot_elyte_c = self.surf_obj.get_net_production_rates(self.elyte_obj) 
        sdot_elyte_o = self.air_elyte_obj.get_net_production_rates(self.elyte_obj)
        
        # Double layer current removes Li from the electrolyte.  Subtract this 
        # from sdot_electrolyte:
        sdot_elyte_c[self.index_Li] -= i_dl / ct.faraday 
            
        # Change in electrolyte species concentration per unit time:
        dCk_elyte_dt = \
            ((sdot_elyte_c * A_surf_ratio + sdot_elyte_o*eps_elyte + self.i_ext_flag * N_k_sep) 
            * self.dyInv / eps_elyte) # first term is reaction second term is seperater? 
        resid[SVptr['C_k_elyte']] = SVdot_loc[SVptr['C_k_elyte']] - dCk_elyte_dt
        #molar production rate of 
        sdot_cath = self.surf_obj.get_net_production_rates(self.product_obj)
        # available interface area on carbon particle
        
        dEpsOxide_dt =  A_avail * np.dot(sdot_cath, self.product_obj.partial_molar_volumes) 
        resid[SVptr['eps_oxide']] = (SVdot_loc[SVptr['eps_oxide']] - dEpsOxide_dt)

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