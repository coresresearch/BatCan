"""
    single_particle_electrode.py

    Class file for single-particle (SPM) electrode methods
"""

import cantera as ct
import numpy as np

class electrode(): 
    """
    Create an electrode object representing the single particle electrode.
    """

    def __init__(self, input_file, inputs, sep_inputs, counter_inputs,    
        electrode_name, params, offset):
        """
        Initialize the model.
        """ 
   
        # Import relevant Cantera objects.
        self.bulk_obj = ct.Solution(input_file, inputs['bulk-phase'])
        self.elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])
        self.conductor_obj = ct.Solution(input_file, inputs['conductor-phase'])
        
        self.surf_obj = ct.Interface(input_file, inputs['surf-phase'], 
            [self.bulk_obj, self.elyte_obj, self.conductor_obj])

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

        # For some models, the elyte thickness is different from that of the 
        # electrode, so we specify is separately:
        self.dy_elyte = self.dy

        # Phase volume fractions
        self.eps_solid = inputs['eps_solid']
        self.eps_elyte = 1 - self.eps_solid

        # Electrode-electrolyte interface area, per unit geometric area.
        # This calculation assumes spherical particles of a single radius, with 
        # no overlap.
        self.A_surf_ratio = (3*self.eps_solid*self.dy/inputs['r_p'])

        # Inverse double layer capacitance, per unit interfacial area.
        self.C_dl_Inv = 1/inputs['C_dl']

        # Microstructure-based transport scaling factor, based on Bruggeman 
        # coefficient of -0.5:
        self.elyte_microstructure = self.eps_elyte**1.5
        
        # SV_offset specifies the index of the first SV variable for the 
        # electode (zero for anode, n_vars_anode + n_vars_sep for the cathode)
        self.SV_offset = offset

        # Determine the electrode capacity (Ah/m2)

        # Max concentration of stored ion (intercalated Li)
        # Save initial X
        X_o = self.bulk_obj.X 
        # Set object concentration to fully lithiated:
        self.bulk_obj.X = inputs['stored-ion']['name']+':1.0' 
        # Concentration of stored Li, per unit volume of intercalation phase:
        Conc = self.bulk_obj[inputs['stored-ion']['name']].concentrations[0]
        
        self.capacity = (Conc*inputs['stored-ion']['charge']*ct.faraday
                *inputs['eps_solid'])*inputs['thickness']/3600

        # Return Cantera object composition to original value:
        self.bulk_obj.X = X_o
        
        # Number of state variables: electrode potential, electrolyte 
        # potential, electrode composition (n_species), electrolyte composition 
        # (n_species)
        self.n_vars = 2 + self.bulk_obj.n_species + self.elyte_obj.n_species

        # This model produces one plot, for the intercalation concentration.
        self.n_plots = 1

        # Set Cantera object state:
        self.bulk_obj.electric_potential = inputs['phi_0']
        # If the user provided an initial composition, use that, here:
        if 'X_0' in inputs:
            self.bulk_obj.TPX = (params['T'], params['P'], inputs['X_0'])
        else:
            self.bulk_obj.TP = params['T'], params['P']

        self.elyte_obj.TP = params['T'], params['P']
        self.surf_obj.TP = params['T'], params['P']
        self.conductor_obj.TP = params['T'], params['P']

        # Set up pointers to specific variables in the solution vector:
        self.SVptr = {}
        self.SVptr['phi_ed'] = np.array([0])
        self.SVptr['phi_dl'] = np.array([1])
        self.SVptr['C_k_ed'] = np.arange(2, 2 + self.bulk_obj.n_species)
        self.SVptr['C_k_elyte'] = np.arange(2 + self.bulk_obj.n_species, 
            2 + self.bulk_obj.n_species + self.elyte_obj.n_species)
        
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
        SV[self.SVptr['C_k_ed']] = self.bulk_obj.concentrations
        SV[self.SVptr['C_k_elyte']] = self.elyte_obj.concentrations

        return SV

    def residual(self, SV, SVdot, sep, counter, params):
        """
        Define the residual for the state of the single particle electrode.

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

        # Read out electrode bulk composition; set the Cantra object:
        C_k_ed = SV_loc[SVptr['C_k_ed']] # Molar density (kmol/m3 of phase)
        X_k_ed = C_k_ed/sum(C_k_ed) # Mole fraction
        self.bulk_obj.X = X_k_ed

        # Set electric potentials for Cantera objects:
        self.bulk_obj.electric_potential = phi_ed
        self.conductor_obj.electric_potential = phi_ed
        self.elyte_obj.electric_potential = phi_elyte
        
        # Faradaic current density is positive when electrons are consumed 
        # (Li transferred to the electrode)
        i_Far = -(ct.faraday 
            * self.surf_obj.get_net_production_rates(self.conductor_obj))
        
        # Calculate the electrolyte species fluxes and associated ionic current 
        # at the boundary with the separator:
        N_k_sep, i_io = sep.electrode_boundary_flux(SV, self, params['T'])

        # Double layer current has the same sign as i_Far, and is based on 
        # charge balance in the electrolyte phase:
        i_dl = self.i_ext_flag*i_io/self.A_surf_ratio - i_Far

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

        # Differential equation for the double layer potential:
        resid[SVptr['phi_dl']] = \
            SVdot_loc[SVptr['phi_dl']] - i_dl*self.C_dl_Inv

        # species production in electrode active material:
        sdot_k_ed = self.surf_obj.get_net_production_rates(self.bulk_obj)

        resid[SVptr['C_k_ed']] = (SVdot_loc[SVptr['C_k_ed']] 
            - self.A_surf_ratio *  sdot_k_ed * self.dyInv / self.eps_solid)

        # Molar production rate of electrode species (kmol/m2/s).
        sdot_elyte = self.surf_obj.get_net_production_rates(self.elyte_obj)
        
        # Double layer current removes Li from the electrolyte.  Subtract this 
        # from sdot_electrolyte:
        sdot_elyte[self.index_Li] -= i_dl / ct.faraday
            
        # Change in electrolyte species concentration per unit time:
        dCk_elyte_dt = \
            ((sdot_elyte * self.A_surf_ratio + self.i_ext_flag * N_k_sep)
            * self.dyInv / self.eps_elyte)
        resid[SVptr['C_k_elyte']] = SVdot_loc[SVptr['C_k_elyte']] - dCk_elyte_dt

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
        C_k_an = solution[2 + self.SV_offset + self.SVptr['C_k_ed'][0],:]
        axs[ax_offset].plot(solution[0,:]/3600, C_k_an)
        axs[ax_offset].set_ylabel(self.name+' Li \n(kmol/m$^3$)')
        axs[ax_offset].set(xlabel='Time (h)')

        return axs
