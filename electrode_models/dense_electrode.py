"""
    dense_electrode.py

    Class file for dense metal (e.g. Li) electrode methods
"""

import cantera as ct
from math import tanh
import numpy as np

class electrode():
    """
    Create an electrode object representing the dense electrode
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
        self.index_Li_elyte = self.elyte_obj.species_index(inputs['mobile-ion'])

        # Electrode thickness
        self.dy = inputs['thickness']
        # The electrode consumption rate quickly goes to zero, below a 
        # user-specified minimum thickness:
        self.min_thickness = inputs['minimum-thickness']

        # Interfacial surface area, per unit geometric area.
        self.A_surf_ratio = inputs['A_surf_ratio']

        # Inverse of the double layer capacitance, per unit interface area:
        self.C_dl_Inv = 1/inputs['C_dl']

        # Thickness of separator node considered as part of the anode domain.  
        # This is "subtracted" off from the total separator thickness.
        self.dy_elyte = inputs['dy_elyte']
        
        # Electrolyte volume fraction in the separator:
        self.eps_elyte = sep_inputs['eps_electrolyte']

        # Microstructure-based transport scaling factor, based on Bruggeman 
        # coefficient of -0.5:
        self.elyte_microstructure = self.eps_elyte**1.5
        
        # SV_offset specifies the index of the first SV variable for the 
        # electode (zero for anode, n_vars_anode + n_vars_sep for the cathode)
        self.SV_offset = offset
        
        # Dense Li is not capacity-limiting, in and of itself.  Rather the 
        # total amount of Li in the system is the limit. This is done in a 
        # separate routine, at a later time. Provide a large placeholder number 
        # here, so that it will not be the minimum, when evaluated later:
        self.capacity = 1e21
            
        # Mumber of state variables: electrode potential, electrolyte 
        # potential, thickness, electrolyte composition (n_species)
        self.n_vars = 3 + self.elyte_obj.n_species

        # This model produces one plot, for the electrode thickness:
        self.n_plots = 1

        # Set the Cantera object state.     
        self.bulk_obj.electric_potential = inputs['phi_0']
        # If the user provided an initial composition, use that, here:
        if 'X_0' in inputs:
            self.bulk_obj.TPX = params['T'], params['P'], inputs['X_0']
        else:
            self.bulk_obj.TP = params['T'], params['P']

        self.elyte_obj.TP = params['T'], params['P']
        self.surf_obj.TP = params['T'], params['P']
        self.conductor_obj.TP = params['T'], params['P']

    def residual(self, t, SV, SVdot, sep, counter, params):
        """
        Define the residual for the state of the dense self.

        This is an array of differential and algebraic governing equations, one for each state variable in the anode (anode plus a thin layer of electrolyte + separator).

        1. The electric potential is an algebraic variable.
            In the anode, phi = 0 is the reference potential for the system.
            In the cathode, the electric potential must be such that the ionic current is spatially in_variant (i.e. it is constant and equal to the external applied current, for galvanostatic simulations).  

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
        
        # Initialize the residual array:
        resid = np.zeros((self.n_vars,))

        # Save local copies of the solution vectors, pointers for this electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]
        SVdot_loc = SVdot[SVptr['electrode']]

        # Read electrode and electrolyte electric potentials:
        phi_ed = SV_loc[SVptr['phi_ed']]
        phi_elyte = phi_ed + SV_loc[SVptr['phi_dl']]

        # Set electric potentials for Cantera objects:
        self.bulk_obj.electric_potential = phi_ed
        self.conductor_obj.electric_potential = phi_ed
        self.elyte_obj.electric_potential = phi_elyte
        
        # Multiplier on the electrode removal reaction. Quickly goes to zero, 
        # for thicknesses below a user-specified minimum:
        mult = tanh(SV_loc[SVptr['thickness']]/self.min_thickness)

        # Molar production rate of electrons in the 'conductor' phase. Electrons are created when lithium is consumed. We scale the the electron creation rate, then, by our multiplier.  When the anode thickness is below the minimum, the electron creation rate goes quickly to zero, but the electron destruction rate is unaffected:
        sdot_electron = \
            (mult*self.surf_obj.get_creation_rates(self.conductor_obj)
            - self.surf_obj.get_destruction_rates(self.conductor_obj))

        # Molar production rate of electrode species (kmol/m2/s). Here, we scale 
        # the destruction rate by our multiplier.
        sdot_electrode = (self.surf_obj.get_creation_rates(self.bulk_obj)
                - mult*self.surf_obj.get_destruction_rates(self.bulk_obj))

        # Faradaic current density is positive when electrons are consumed 
        # (Li transferred to the anode)
        i_Far = -ct.faraday*sdot_electron

        # Flux of electrolyte species between the separator and the electrolyte 
        # in the current electrode domain:
        N_k_sep, i_io = sep.electrode_boundary_flux(SV, self, params['T'])

        # Electrode electric potential
        if self.name=='anode':
            # For the anode, the electric potential is an algebraic variable, 
            # always equal to zero:
            resid[SVptr['phi_ed']] = SV_loc[SVptr['phi_ed']]

        elif self.name=='cathode':
            # For the cathode, the potential of the cathode must be such that 
            # the electrolyte electric potential (calculated as phi_ca + 
            # dphi_dl) produces the correct ionic current between the separator # and cathode:
            if params['boundary'] == 'current':
                resid[SVptr['phi_ed']] = i_io - params['i_ext']
            elif params['boundary'] == 'potential':
                # Potential at time t:
                phi = np.interp(t, params['times'], params['potentials'])

                # Cell potential must equal phi:
                resid[SVptr['phi_ed']] = SV_loc[SVptr['phi_ed']] - phi
        
        # Double layer current has the same sign as i_Far, and is based on 
        # charge balance in the electrolyte phase:
        i_dl = self.i_ext_flag*i_io/self.A_surf_ratio - i_Far
        
        # Differential equation for the double layer potential difference:
        resid[SVptr['phi_dl']] = (SVdot_loc[SVptr['phi_dl']] 
            - i_dl*self.C_dl_Inv)
        
        # Change in thickness per time:
        dH_dt = np.dot(sdot_electrode, self.bulk_obj.partial_molar_volumes)
        resid[SVptr['thickness']] = SVdot_loc[SVptr['thickness']] - dH_dt
        
        # Set time derivatives for electrolyte species concentrations to zero 
        # (temporary)

        # Molar production rate of electrode species (kmol/m2/s). Here, we scale 
        # the destruction rate by our multiplier.
        sdot_electrolyte = \
            (mult*self.surf_obj.get_creation_rates(self.elyte_obj)
            - self.surf_obj.get_destruction_rates(self.elyte_obj))

        # Double layer current removes Li from the electrolyte.  Add this to 
        # sdot_electrolyte:
        sdot_electrolyte[self.index_Li_elyte] -= i_dl / ct.faraday
        dCk_elyte_dt = \
            (sdot_electrolyte * self.A_surf_ratio 
            + self.i_ext_flag * N_k_sep) / self.dy_elyte 
        resid[SVptr['C_k_elyte']] = SVdot_loc[SVptr['C_k_elyte']] - dCk_elyte_dt

        return resid

    def initialize(self, inputs, sep_inputs):

        # Initialize the solution vector for the electrode domain:
        SV = np.zeros([self.n_vars])

        # Set up pointers to specific variables in the solution vector:
        self.SVptr = {}
        self.SVptr['phi_ed'] = np.array([0])
        self.SVptr['phi_dl'] = np.array([1])
        self.SVptr['thickness'] = np.array([2])
        self.SVptr['C_k_elyte'] = np.arange(3, 
                3 + self.elyte_obj.n_species)

        self.SVnames = (['phi_ed', 'phi_dl', 'thickness'] 
            + self.elyte_obj.species_names[:])
            
        # There is only one node, but give the pointer a shape so that SVptr
        # ['C_k_elyte'][j] accesses the pointer array:
        self.SVptr['C_k_elyte'].shape = (1, self.elyte_obj.n_species)

        # A pointer to where the SV variables for this electrode are, within 
        # the overall solution vector for the entire problem:
        self.SVptr['electrode'] = np.arange(self.SV_offset, 
            self.SV_offset+self.n_vars)

        # Save the SV indices of any algebraic variables:
        self.algvars = self.SV_offset + self.SVptr['phi_ed'][:]
        
        # Load intial state variable values:
        SV[self.SVptr['phi_ed']] = inputs['phi_0']
        SV[self.SVptr['phi_dl']] = sep_inputs['phi_0'] - inputs['phi_0']
        SV[self.SVptr['thickness']] = inputs['thickness']
        SV[self.SVptr['C_k_elyte']] = self.elyte_obj.concentrations

        return SV

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

    def species_lim(self, SV, val):
        """
        Check to see if the minimum species concentration limit has been exceeded.
        """
        # Save local copies of the solution vector and pointers for this electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]

        # For each electrode point, find the minimum species concentration, and # compare to the user-provided minimum.  Save only the minimum value:
        species_eval = min(SV_loc[SVptr['C_k_elyte'][0]]) - val
        
        # The simulation  looks for instances where this value changes sign 
        # (i.e. where it crosses zero)    
        return species_eval

    def adjust_separator(self, sep):
        """
        The electrode domain considers the electrode object plus a thin layer of the separator, adjacent to the self. We subtract this thickness from the total separator thickness, so that we do not inadvertently increase the total transport resistance through the separator.
        """
        sep.dy -= self.dy_elyte
        
        # Reduce the number of points in the separator by one, unless the 
        # separator already only contains one point (which is the case for the 
        # `ionic_resistor` model. In this case, leave sep.n_points at 1.)
        sep.n_points = max(sep.n_points - 1, 1)
        
        return sep

    def output(self, axs, solution, SV_offset, ax_offset):

        axs[ax_offset].plot(solution[0,:]/3600, 
            1e6*solution[SV_offset+int(self.SVptr['thickness'])])
        axs[ax_offset].set_ylabel(self.name+' Thickness \n($\mu$m)')

        return axs
