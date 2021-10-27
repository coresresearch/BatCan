"""
    metal_air_single_particle.py

    Class file for metal air electrode methods
"""

import cantera as ct
from math import tanh
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
        self.gas_elyte_obj = ct.Interface(input_file, inputs['elyte-iphase'], [self.gas_obj, self.elyte_obj])
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
        self.N_y = inputs['n-points'] # change to n-points
        self.dy = inputs['thickness']/self.N_y
        self.dyInv = 1/self.dy

        # Phase volume fractions
        self.eps_host = inputs['eps_host']
        self.eps_product_init =inputs['eps_product']
        self.eps_elyte_init = 1 - self.eps_host - self.eps_product_init

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

        # Max voume concentration of the product species (assuming all 
        # electrolyte has been replaced by oxide)
        stored_species = inputs['stored-species']
        v_molar_prod = \
            self.product_obj[stored_species['name']].partial_molar_volumes[0]

        self.capacity = (stored_species['charge']*ct.faraday
                * self.eps_elyte_init * inputs['thickness']
                / (3600 * v_molar_prod))
                 
        # Minimum volume fraction for the product phase, below which product 
        # phase consumption reaction shut off:
        self.product_phase_min = inputs['product-phase-min']
        # Number of state variables: electrode potential, electrolyte composition, oxide volume fraction 
        self.n_vars = 3 + self.elyte_obj.n_species
        self.n_vars_tot = self.N_y*self.n_vars

        # This model produces zero plots, but might someday.
        self.n_plots = 2

        # Store any extra species to be ploted
        self.plot_species = []
        [self.plot_species.append(sp['name']) for sp in inputs['plot-species']]

        # Set Cantera object state:
        self.host_obj.TP = params['T'], params['P']
        self.elyte_obj.TP = params['T'], params['P']
        self.surf_obj.TP = params['T'], params['P']
        #self.conductor_obj.TP = params['T'], params['P']

        # Set up pointers to specific variables in the solution vector:
        self.SVptr = {}
        self.SVptr['phi_ed'] = np.arange(0, self.n_vars_tot, self.n_vars)
        self.SVptr['phi_dl'] = np.arange(1, self.n_vars_tot, self.n_vars)
        self.SVptr['eps_product'] = np.arange(2, self.n_vars_tot, self.n_vars)
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
        SV[self.SVptr['phi_ed']] = inputs['phi_0'] #v
        SV[self.SVptr['phi_dl']] = sep_inputs['phi_0'] - inputs['phi_0'] #V
        SV[self.SVptr['eps_product']] = self.eps_product_init #Volume Fraction
        SV[self.SVptr['C_k_elyte']] = self.elyte_obj.concentrations #
        
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
       
        j = 0
        # Read out properties:
        phi_ed = SV_loc[SVptr['phi_ed'][j]]
        phi_elyte = phi_ed + SV_loc[SVptr['phi_dl'][j]]
        # print('phi_elyte = ', phi_elyte)
        c_k_elyte = SV_loc[SVptr['C_k_elyte'][j]]
        eps_product = SV_loc[SVptr['eps_product'][j]]
         # Set Cantera object properties:
        self.host_obj.electric_potential = phi_ed
        self.elyte_obj.electric_potential = phi_elyte
        self.elyte_obj.X = c_k_elyte
        # Set microstructure multiplier for effective diffusivities
        eps_elyte = 1. - eps_product - self.eps_host
        self.elyte_microstructure = eps_elyte**1.5
        # Read electrolyte fluxes at the separator boundary:
        N_k_sep, i_io_sep = sep.electrode_boundary_flux(SV, self, params['T'])
               
        #calculate flux out    
        FoRT = ct.faraday/ct.gas_constant/self.elyte_obj.T  
        phi_ed_next = SV_loc[SVptr['phi_ed'][j+1]]
        phi_elyte_next = phi_ed_next +SV_loc[SVptr['phi_dl'][j+1]]
        # print('phi_elyte = ', phi_elyte)
        c_k_elyte_next = SV_loc[SVptr['C_k_elyte'][j+1]]
        eps_product_next = SV_loc[SVptr['eps_product'][j+1]]
        eps_elyte_next = 1. - eps_product_next- self.eps_host
        
        #FIX: Actual diffusion  Question: best way to pull this 
        D_k_temp = np.array([1.11e-10, 6.98e-11, 8.79e-11, 4.26e-11, 2e-13]) #sep.D_k?
        C_k_elyte_int = 0.5*(c_k_elyte + c_k_elyte_next)
        eps_int = 0.5*(eps_elyte + eps_elyte_next)
        D_k_elyte = D_k_temp*eps_int**(1.5)
        D_k_elyte_mig = D_k_elyte*self.elyte_obj.charges*FoRT*C_k_elyte_int/eps_int #changed this?

        N_k = (D_k_elyte*(c_k_elyte/eps_elyte - c_k_elyte_next/eps_elyte_next)+ \
            D_k_elyte_mig*(phi_elyte - phi_elyte_next))*self.dyInv

        if self.name=='anode':
            # The electric potential of the anode = 0 V.
            resid[[SVptr['phi_ed'][j]]] = phi_ed

        elif self.name=='cathode':
            # For the cathode, the potential of the cathode must be such that 
            # the electrolyte electric potential (calculated as phi_ca + 
            # dphi_dl) produces the correct ionic current between the separator # and cathode:
            if params['boundary'] == 'current':
                resid[SVptr['phi_ed'][j]] = i_io_sep - params['i_ext']
            elif params['boundary'] == 'potential':                  
                resid[SVptr['phi_ed'][j]] = (SV_loc[SVptr['phi_ed']] 
                    - params['potential']) 
        
        # Calculate available surface area (m2 interface per m3 electrode):
        A_avail = self.A_init - eps_product/self.th_oxide
        # Convert to m2 interface per m2 geometric area:
        A_surf_ratio = A_avail*self.dy
        # Multiplier to scale phase destruction rates.  As eps_product drops 
        # below the user-specified minimum, any reactions that consume the 
        # phase have their rates quickly go to zero:
        mult = tanh(eps_product / self.product_phase_min)

        # Chemical production rate of the product phase: (mol/m2 interface/s)
        sdot_product = (self.surf_obj.get_creation_rates(self.product_obj)
            - mult * self.surf_obj.get_destruction_rates(self.product_obj))
      
        # Rate of change of the product phase volume fraction:
        resid[SVptr['eps_product'][j]] = (SVdot_loc[SVptr['eps_product'][j]] 
            - A_avail * np.dot(sdot_product, self.product_obj.partial_molar_volumes))

        # Production rate of the electron (moles / m2 interface / s)
        sdot_electron = (mult * self.surf_obj.get_creation_rates(self.host_obj)
            - self.surf_obj.get_destruction_rates(self.host_obj))

        # Positive Faradaic current corresponds to positive charge created in 
        # the electrode:
        i_Far = -(ct.faraday * sdot_electron)

        # Double layer current has the same sign as i_Far
        i_dl = self.i_ext_flag*(i_io_sep)/A_surf_ratio - i_Far
        resid[SVptr['phi_dl'][j]] = SVdot_loc[SVptr['phi_dl'][j]] - i_dl*self.C_dl_Inv
        #change in concentration
        sdot_elyte_host = (mult*self.surf_obj.get_creation_rates(self.elyte_obj)
            - self.surf_obj.get_destruction_rates(self.elyte_obj))
        sdot_elyte_host[self.index_Li] -= i_dl / ct.faraday 
        
        # print(sdot_product, sdot_elyte_host)
        resid[SVptr['C_k_elyte'][j]] = (SVdot_loc[SVptr['C_k_elyte'][j]] 
            - (N_k_sep - N_k + sdot_elyte_host * A_surf_ratio) 
            * self.dyInv)
        j = 1
                # Read out properties:
        phi_ed = phi_ed_next
        phi_elyte = phi_elyte_next
        # print('phi_elyte = ', phi_elyte)
        c_k_elyte = c_k_elyte_next
        eps_product = eps_product_next
         # Set Cantera object properties:
        self.host_obj.electric_potential = phi_ed
        self.elyte_obj.electric_potential = phi_elyte
        self.elyte_obj.X = c_k_elyte
        # Set microstructure multiplier for effective diffusivities
        eps_elyte = eps_elyte_next

        N_k_int = N_k
    
        #calculate flux out    
        FoRT = ct.faraday/ct.gas_constant/self.elyte_obj.T  
        phi_ed_next = SV_loc[SVptr['phi_ed'][j+1]]
        phi_elyte_next = phi_ed + SV_loc[SVptr['phi_dl'][j+1]]
        # print('phi_elyte = ', phi_elyte)
        c_k_elyte_next = SV_loc[SVptr['C_k_elyte'][j+1]]
        eps_product_next = SV_loc[SVptr['eps_product'][j+1]]
        eps_elyte_next = 1. - eps_product_next- self.eps_host
        
        C_k_elyte_int = 0.5*(c_k_elyte + c_k_elyte_next)
        eps_int = 0.5*(eps_elyte + eps_elyte_next)
        D_k_elyte = D_k_temp*eps_int**(1.5)
        D_k_elyte_mig = D_k_elyte*self.elyte_obj.charges*FoRT*C_k_elyte_int

        N_k = (D_k_elyte*(c_k_elyte/eps_elyte - c_k_elyte_next/eps_elyte_next)+ \
            D_k_elyte_mig*(phi_elyte - phi_elyte_next))*self.dyInv

        if self.name=='anode':
            # The electric potential of the anode = 0 V.
            resid[[SVptr['phi_ed'][j]]] = phi_ed

        elif self.name=='cathode':
            # For the cathode, the potential of the cathode must be such that 
            # the electrolyte electric potential (calculated as phi_ca + 
            # dphi_dl) produces the correct ionic current between the separator # and cathode:
            if params['boundary'] == 'current':
                resid[SVptr['phi_ed'][j]] = i_io_sep - params['i_ext']
            elif params['boundary'] == 'potential':                  
                resid[SVptr['phi_ed'][j]] = (SV_loc[SVptr['phi_ed']] 
                    - params['potential']) 
        
        # Calculate available surface area (m2 interface per m3 electrode):
        A_avail = self.A_init - eps_product/self.th_oxide
        # Convert to m2 interface per m2 geometric area:
        A_surf_ratio = A_avail*self.dy
        # Multiplier to scale phase destruction rates.  As eps_product drops 
        # below the user-specified minimum, any reactions that consume the 
        # phase have their rates quickly go to zero:
        mult = tanh(eps_product / self.product_phase_min)

        # Chemical production rate of the product phase: (mol/m2 interface/s)
        sdot_product = (self.surf_obj.get_creation_rates(self.product_obj)
            - mult * self.surf_obj.get_destruction_rates(self.product_obj))
      
        # Rate of change of the product phase volume fraction:
        resid[SVptr['eps_product'][j]] = (SVdot_loc[SVptr['eps_product'][j]] 
            - A_avail * np.dot(sdot_product, self.product_obj.partial_molar_volumes))

        # Production rate of the electron (moles / m2 interface / s)
        sdot_electron = (mult * self.surf_obj.get_creation_rates(self.host_obj)
            - self.surf_obj.get_destruction_rates(self.host_obj))

        # Positive Faradaic current corresponds to positive charge created in 
        # the electrode:
        i_Far = -(ct.faraday * sdot_electron)

        # Double layer current has the same sign as i_Far
        i_dl = self.i_ext_flag*(i_io_sep)/A_surf_ratio - i_Far
        resid[SVptr['phi_dl'][j]] = SVdot_loc[SVptr['phi_dl'][j]] - i_dl*self.C_dl_Inv
        #change in concentration
        sdot_elyte_host = (mult*self.surf_obj.get_creation_rates(self.elyte_obj)
            - self.surf_obj.get_destruction_rates(self.elyte_obj))
        sdot_elyte_host[self.index_Li] -= i_dl / ct.faraday 

        resid[SVptr['C_k_elyte'][j]] = (SVdot_loc[SVptr['C_k_elyte'][j]] 
            - (N_k_int - N_k + sdot_elyte_host * A_surf_ratio) 
            * self.dyInv)
        j = 2
        phi_ed = phi_ed_next
        phi_elyte = phi_elyte_next
        # print('phi_elyte = ', phi_elyte)
        c_k_elyte = c_k_elyte_next
        eps_product = eps_product_next
         # Set Cantera object properties:
        self.host_obj.electric_potential = phi_ed
        self.elyte_obj.electric_potential = phi_elyte
        self.elyte_obj.X = c_k_elyte
        # Set microstructure multiplier for effective diffusivities
        eps_elyte = eps_elyte_next
        # Molar production rate of electrolyte species at the electrolyte-air 
        # interface (kmol / m2 of interface / s) 
        if self.name=='anode':
            # The electric potential of the anode = 0 V.
            resid[[SVptr['phi_ed'][j]]] = phi_ed

        elif self.name=='cathode':
            # For the cathode, the potential of the cathode must be such that 
            # the electrolyte electric potential (calculated as phi_ca + 
            # dphi_dl) produces the correct ionic current between the separator # and cathode:
            if params['boundary'] == 'current':
                resid[SVptr['phi_ed'][j]] = i_io_sep - params['i_ext']
            elif params['boundary'] == 'potential':                  
                resid[SVptr['phi_ed'][j]] = (SV_loc[SVptr['phi_ed']] 
                    - params['potential']) 
        
        # Calculate available surface area (m2 interface per m3 electrode):
        A_avail = self.A_init - eps_product/self.th_oxide
        # Convert to m2 interface per m2 geometric area:
        A_surf_ratio = A_avail*self.dy
        # Multiplier to scale phase destruction rates.  As eps_product drops 
        # below the user-specified minimum, any reactions that consume the 
        # phase have their rates quickly go to zero:
        mult = tanh(eps_product / self.product_phase_min)

        # Chemical production rate of the product phase: (mol/m2 interface/s)
        sdot_product = (self.surf_obj.get_creation_rates(self.product_obj)
            - mult * self.surf_obj.get_destruction_rates(self.product_obj))
      
        # Rate of change of the product phase volume fraction:
        resid[SVptr['eps_product'][j]] = (SVdot_loc[SVptr['eps_product'][j]] 
            - A_avail * np.dot(sdot_product, self.product_obj.partial_molar_volumes))

        # Production rate of the electron (moles / m2 interface / s)
        sdot_electron = (mult * self.surf_obj.get_creation_rates(self.host_obj)
            - self.surf_obj.get_destruction_rates(self.host_obj))

        # Positive Faradaic current corresponds to positive charge created in 
        # the electrode:
        i_Far = -(ct.faraday * sdot_electron)

        # Double layer current has the same sign as i_Far
        i_dl = self.i_ext_flag*(i_io_sep)/A_surf_ratio - i_Far
        resid[SVptr['phi_dl'][j]] = SVdot_loc[SVptr['phi_dl'][j]] - i_dl*self.C_dl_Inv
        #change in concentration
        sdot_elyte_host = (mult*self.surf_obj.get_creation_rates(self.elyte_obj)
            - self.surf_obj.get_destruction_rates(self.elyte_obj))
        sdot_elyte_host[self.index_Li] -= i_dl / ct.faraday 
        sdot_elyte_air = \
            self.gas_elyte_obj.get_net_production_rates(self.elyte_obj) 
        resid[SVptr['C_k_elyte'][j]] = (SVdot_loc[SVptr['C_k_elyte'][j]] 
            - (N_k + sdot_elyte_air + sdot_elyte_host * A_surf_ratio) 
            * self.dyInv)
        return resid
        
    def voltage_lim(self, SV, val):
        """
        Check to see if the voltage limits have been exceeded.
        """
        # Save local copies of the solution vector and pointers for this electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]
        
        # Calculate the current voltage, relative to the limit.  The simulation 
        # looks for instances where this value changes sign (i.e. where it 
        # crosses zero)    
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
        eps_product_ptr = (2 + self.SV_offset + self.SVptr['eps_product'][0])
        
        axs[ax_offset].plot(solution[0,:]/3600, solution[eps_product_ptr, :])
        axs[ax_offset].set_ylabel(self.name+' product \n volume fraction')

        for name in self.plot_species:
            species_ptr = self.elyte_obj.species_index(name)
            C_k_elyte_ptr = (2 + self.SV_offset 
                + self.SVptr['C_k_elyte'][0, species_ptr])
            axs[ax_offset+1].plot(solution[0,:]/3600, 
                1000*solution[C_k_elyte_ptr,:])

        axs[ax_offset+1].legend(self.plot_species)
        axs[ax_offset+1].set_ylabel('Elyte Species Conc. \n (mol m$^{-3}$)')
        return axs

#Official Soundtrack:
    #Cursive - Happy Hollow
    #Japancakes - If I Could See Dallas
    #Jimmy Eat World - Chase the Light + Invented
    #Lay - Lit
    #George Ezra - Staying at Tamara's