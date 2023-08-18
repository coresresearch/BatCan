"""
    air_electrode.py

    Class file for air electrode methods
"""

import cantera as ct
from math import tanh
import numpy as np

class electrode():
    """
    Create an electrode object representing the air electrode.
    """

    def __init__(self, input_file, inputs, sep_inputs, counter_inputs,
        electrode_name, params, offset):
        """
        Initialize the model.
        """

        # Import relevant Cantera objects.
        self.gas_obj = ct.Solution(input_file, inputs['gas-phase'])
        self.elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])
        self.gas_elyte_obj = ct.Interface(input_file, inputs['elyte-iphase'],
            [self.gas_obj, self.elyte_obj])
        self.host_obj = ct.Solution(input_file, inputs['host-phase'])
        self.product_obj = ct.Solution(input_file, inputs['product-phase'])
        self.surf_obj = ct.Interface(input_file, inputs['surf-iphase'],
            [self.product_obj, self.elyte_obj, self.host_obj])


        # Electrode thickness and inverse thickness:
        self.n_points = inputs['n-points']
        self.dy = inputs['thickness']/self.n_points
        self.dyInv = 1/self.dy

        # Anode or cathode? Positive external current delivers positive charge
        # to the anode, and removes positive charge from the cathode. For the
        # cathode, the first node is at the separator (j=0), whereas the last
        # node has j=0.
        self.name = electrode_name
        if self.name=='anode':
            self.i_ext_flag = -1
            self.nodes = list(range(self.n_points-1,-1,-1))
        elif self.name=='cathode':
            self.i_ext_flag = 1
            self.nodes = list(range(self.n_points))
        else:
            raise ValueError("Electrode must be an anode or a cathode.")

        # Store the species index of the Li ion in the Cantera object for the
        # electrolyte phase:
        self.index_Li_elyte = self.elyte_obj.species_index(inputs['mobile-ion'])
        eps_host = inputs['eps_host']

        try:
            count = len(eps_host)
            if count == self.n_points:
                self.eps_host = np.asarray(eps_host)
            elif count == 1:
                self.eps_host = np.repeat(eps_host, self.n_points)
            else:
               raise ValueError("Porosity must be either a single scalar, or"
                " must match the discretization of the air electrode.") 
        except:
            float(eps_host)
            self.eps_host = np.repeat(eps_host, self.n_points)

        self.eps_product_init = inputs['eps_product']
        self.eps_elyte_init = 1. - self.eps_host - self.eps_product_init

        self.sigma_el =inputs['sigma_el']
        # The following calculations assume spherical particles of a single
        # radius, with no overlap.
        self.r_host = inputs['r_host']
        self.th_product = inputs['th_product']
        self.V_host = 4./3. * np.pi * (self.r_host)**3  # Volume of a single carbon / host particle [m3]
        self.A_host = 4. * np.pi * (self.r_host)**2 # Surface area of a single carbon / host particle [m2]
        # m2 of host-electrolyte interface / m3 of total volume [m^-1]
        if 'host_form' not in inputs or inputs['host_form'] ==  'sphere':
            self.A_init = self.eps_host * 3./self.r_host
        elif inputs['host_form'] == 'cylinder':
            self.A_init = self.eps_host * 2./self.r_host
        else:
            raise ValueError("Support host_form values include: 'sphere'",
                " or 'cylinder'.")

        if 'host_surf_frac' in inputs:
            self.A_init *= inputs['host_surf_frac']
        # For some models, the elyte thickness is different from that of the
        # electrode, so we specify it separately:
        self.dy_elyte = self.dy

        # Inverse double layer capacitance, per unit interfacial area.
        self.C_dl_Inv = 1/inputs['C_dl']

        # Microstructure-based transport scaling factor, based on Bruggeman
        # coefficient of -0.5:
        self.elyte_microstructure = self.eps_elyte_init[0]**1.5 
        #This is used by the electrode_boundary_flux
        # SV_offset specifies the index of the first SV variable for the
        # electode (zero for anode, n_vars_anode + n_vars_sep for the cathode)
        self.SV_offset = offset

        # Determine the electrode capacity (Ah/m2)

        # Max voume concentration of the product species (assuming all
        # electrolyte has been replaced by product species)
        stored_species = inputs['stored-species']
        v_molar_prod = \
            self.product_obj[stored_species['name']].partial_molar_volumes[0]

        self.capacity = (stored_species['charge']*ct.faraday
                * (np.average(self.eps_elyte_init))* inputs['thickness']
                / (3600 * v_molar_prod))

        # Minimum volume fraction for the product phase, below which product
        #   phase consumption reaction shut off:
        self.product_phase_min = inputs['product-phase-min']
        # Number of state variables: electrode potential, double layer
        #   potential, electrolyte composition, product phase volume fraction
        self.n_vars = 3 + self.elyte_obj.n_species
        self.n_vars_tot = self.n_points*self.n_vars

        # Specify the number of plots
        #   1 - Elyte species concentrations for select species
        #   2 - Cathode produce phase volume fraction
        self.n_plots = 2

        # Store any extra species to be ploted
        self.plot_species = []
        [self.plot_species.append(sp['name']) for sp in inputs['plot-species']]

        # Set Cantera object state:
        self.gas_obj.TP = params['T'], params['P']
        self.gas_elyte_obj.TP = params['T'], params['P']
        self.host_obj.TP = params['T'], params['P']
        self.elyte_obj.TP = params['T'], params['P']
        self.surf_obj.TP = params['T'], params['P']

    def initialize(self, inputs, sep_inputs):

        # Set up pointers to the variables in the solution vector:
        self.SVptr = {}
        self.SVptr['phi_ed'] = np.arange(0, self.n_vars_tot, self.n_vars)
        self.SVptr['phi_dl'] = np.arange(1, self.n_vars_tot, self.n_vars)
        self.SVptr['eps_product'] = np.arange(2, self.n_vars_tot, self.n_vars)
        self.SVptr['C_k_elyte'] = np.ndarray(shape=(self.n_points,
            self.elyte_obj.n_species), dtype='int')
        for i in range(self.n_points):
            self.SVptr['C_k_elyte'][i,:] = range(3 + i*self.n_vars,
                3 + i*self.n_vars + self.elyte_obj.n_species)

        self.SVnames = (['phi_ed', 'phi_dl', 'eps_product']
            + self.elyte_obj.species_names[:])*self.n_points

        # A pointer to where the SV variables for this electrode are, within
        # the overall solution vector for the entire problem:
        self.SVptr['electrode'] = np.arange(self.SV_offset,
            self.SV_offset+self.n_vars_tot)

        # Save the indices of any algebraic variables:
        self.algvars = self.SV_offset + self.SVptr['phi_ed'][:]

        # Initialize the solution vector for the electrode domain:
        SV = np.zeros(self.n_vars_tot)

        # Load intial state variables: Change it later
        SV[self.SVptr['phi_ed']] = inputs['phi_0'] # V
        SV[self.SVptr['phi_dl']] = sep_inputs['phi_0'] - inputs['phi_0'] #V
        SV[self.SVptr['eps_product']] = self.eps_product_init #Volume Fraction
        SV[self.SVptr['C_k_elyte']] = self.elyte_obj.concentrations # kmol/m3

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

        Sign convention:
            - For either electrode, the sign convention assumes that positive flux is from the separator toward the current collector.  All loops over the finite volumes proceeds in this same direction.
        """
        # Initialize the residual:
        resid = np.zeros((self.n_vars_tot,))

        # Save local copies of the solution vectors, pointers for this
        #   electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]
        SVdot_loc = SVdot[SVptr['electrode']]

        # Start at the separator boundary:
        j = self.nodes[0]

        # Read out properties:
        phi_ed = SV_loc[SVptr['phi_ed'][j]]
        phi_elyte = phi_ed + SV_loc[SVptr['phi_dl'][j]]
        c_k_elyte = SV_loc[SVptr['C_k_elyte'][j]]
        eps_product = SV_loc[SVptr['eps_product'][j]]
        eps_elyte = 1. - eps_product - self.eps_host[j]
        self.elyte_microstructure = eps_elyte**1.5
        # Read electrolyte fluxes at the separator boundary.  No matter the
        # electrode, the function returns a value where flux to the electrode
        # is considered positive. We multiply by `i_ext_flag` to get the
        # correct sign.
        N_k_in, i_io_in = (self.i_ext_flag*X for X in
            sep.electrode_boundary_flux(SV, self, params['T']))

        # No electronic current at the separator boundary:
        i_el_in = 0

        # Loop through the finite volumes:
        for j_next in self.nodes[1:]:
            # Set Cantera object properties:
            self.host_obj.electric_potential = phi_ed
            self.elyte_obj.electric_potential = phi_elyte
            try:
                self.elyte_obj.X = c_k_elyte/sum(c_k_elyte)
            except:
                X_elyte = params['species-default']
                self.elyte_obj.X = X_elyte


            # Set microstructure multiplier for effective diffusivities
            #TODO #48
            self.elyte_microstructure = eps_elyte**1.5

            # Read out state variables for 'next' node toward air boundary:
            phi_ed_next = SV_loc[SVptr['phi_ed'][j_next]]
            phi_elyte_next = phi_ed_next + SV_loc[SVptr['phi_dl'][j_next]]
            c_k_elyte_next = SV_loc[SVptr['C_k_elyte'][j_next]]
            eps_product_next = SV_loc[SVptr['eps_product'][j_next]]
            eps_elyte_next = 1. - eps_product_next- self.eps_host[j_next]

            # Load the node properties into dict structures
            # (1 = local, 2 = next node)
            state_1 = {'C_k': c_k_elyte, 'phi':phi_elyte,
                'T': params['T'], 'dy':self.dy, 'microstructure':eps_elyte**1.5}
            state_2 = {'C_k': c_k_elyte_next, 'phi':phi_elyte_next,
                'T': params['T'], 'dy':self.dy,
                'microstructure':eps_elyte_next**1.5}

            # Calculate fluxes and currents out of this node, into the next
            # node toward the current collector:
            N_k_out, i_io_out = sep.elyte_transport(state_1, state_2, sep)
            i_el_out = (self.sigma_el*(phi_ed - phi_ed_next)*self.dyInv)

            # Total current (ionic plus electronic) must be spatially invariant.
            resid[SVptr['phi_ed'][j]] = i_io_in - i_io_out + i_el_in - i_el_out

            # Calculate available surface area (m2 interface per m3 electrode):
            A_avail = self.A_init[j] - eps_product/self.th_product
            # Convert to m2 interface per m2 geometric area:
            A_surf_ratio = A_avail*self.dy
            # Multiplier to scale phase destruction rates.  As eps_product
            # drops below the user-specified minimum, any reactions that
            # consume the phase have their rates quickly go to zero:
            mult = tanh(eps_product / self.product_phase_min)

            # Chemical production rate of the product phase:
            # (kmol/m2-interface/s)
            sdot_product = (self.surf_obj.get_creation_rates(self.product_obj)
                - mult * self.surf_obj.get_destruction_rates(self.product_obj))

            # Rate of change of the product phase volume fraction:
            resid[SVptr['eps_product'][j]] = \
                (SVdot_loc[SVptr['eps_product'][j]] - A_avail
                * np.dot(sdot_product, self.product_obj.partial_molar_volumes))

            # Production rate of the electron (moles / m2 interface / s)
            sdot_electron = \
                (mult * self.surf_obj.get_creation_rates(self.host_obj)
                - self.surf_obj.get_destruction_rates(self.host_obj))

            # Positive Faradaic current corresponds to positive charge created
            # in the electrode:
            i_Far = -(ct.faraday * sdot_electron)

            # Double layer current has the same sign as i_Far
            i_dl = self.i_ext_flag*(i_io_in - i_io_out)/A_surf_ratio - i_Far
            resid[SVptr['phi_dl'][j]] = (SVdot_loc[SVptr['phi_dl'][j]] -
                i_dl*self.C_dl_Inv)

            # Species production rate for electrolyte species:
            sdot_elyte_host = \
                (mult*self.surf_obj.get_creation_rates(self.elyte_obj)
                - self.surf_obj.get_destruction_rates(self.elyte_obj))

            # The double layer current acts as an additional chemical source/
            # sink term:
            sdot_elyte_host[self.index_Li_elyte] -= i_dl / ct.faraday

            # Change in electrolyte species concentration, per unit time:
            resid[SVptr['C_k_elyte'][j]] = (SVdot_loc[SVptr['C_k_elyte'][j]]
                - (N_k_in - N_k_out+ sdot_elyte_host * A_surf_ratio)
                * self.dyInv)/eps_elyte

            # Re-set "next" node state variables to be the new "current" node
            # state variables:
            phi_ed = phi_ed_next
            phi_elyte = phi_elyte_next
            c_k_elyte = c_k_elyte_next
            eps_product = eps_product_next
            eps_elyte = eps_elyte_next

            # Similarly, all fluxes out become fluxes in, when we move to the
            # next node:
            N_k_in = N_k_out
            i_io_in = i_io_out
            i_el_in = i_el_out
            j = j_next

        """ The final node is at the air boundary: """
        # No ionic current or electrolyte flux to the air:
        i_io_out = 0
        N_k_out = np.zeros_like(N_k_in)

        # Set Cantera object properties:
        self.host_obj.electric_potential = phi_ed
        self.elyte_obj.electric_potential = phi_elyte
        try:
            self.elyte_obj.X = c_k_elyte/sum(c_k_elyte)
        except:
            if 'species-default' in params:
                self.elyte_obj.X = params['species-default']
            else:
                pass

        # Electric potential boundary condition:
        if self.name=='anode':
            # The electric potential of the anode = 0 V.
            resid[[SVptr['phi_ed'][j]]] = phi_ed

        elif self.name=='cathode':
            if params['boundary'] == 'current':
                # Total current (i_io + i_el) into the node equals i_ext:
                i_el_out = params['i_ext']
                resid[SVptr['phi_ed'][j]] =  i_io_in + i_el_in - i_el_out
            elif params['boundary'] == 'potential':
                # Electrode potential = cell potential:
                resid[SVptr['phi_ed'][j]] = (SV_loc[SVptr['phi_ed']]
                    - params['potential'])

        # Calculate available surface area (m2 interface per m3 electrode):
        A_avail = self.A_init[j] - eps_product/self.th_product
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
        i_dl = self.i_ext_flag*(i_io_in - i_io_out)/A_surf_ratio - i_Far
        resid[SVptr['phi_dl'][j]] = (SVdot_loc[SVptr['phi_dl'][j]]
            - i_dl*self.C_dl_Inv)


        # Molar production rate of electrolyte species at the electrolyte-electrode interface (kmol / m2 of interface / s)
        sdot_elyte_host = (mult*self.surf_obj.get_creation_rates(self.elyte_obj)
            - self.surf_obj.get_destruction_rates(self.elyte_obj))

        # Double layer current represents an additional chemical source/sink
        # term, for electrolyte chemical species:
        sdot_elyte_host[self.index_Li_elyte] -= i_dl / ct.faraday

        # Molar production rate of electrolyte species at the electrolyte-air
        # interface (kmol / m2 of interface / s)
        sdot_elyte_air = \
            self.gas_elyte_obj.get_net_production_rates(self.elyte_obj)

        # Rate of change of electrolyte chemical species molar concentration:
        resid[SVptr['C_k_elyte'][j]] = (SVdot_loc[SVptr['C_k_elyte'][j]]
            - (N_k_in - N_k_out + sdot_elyte_air
            + sdot_elyte_host * A_surf_ratio)
            * self.dyInv)/eps_elyte

        eps_elyte = 1. - eps_product - self.eps_host[0]
        self.elyte_microstructure = eps_elyte**1.5

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
        voltage_eval = SV_loc[SVptr['phi_ed'][-1]] - val

        # if voltage_eval <= 0.:
        #     print("Voltage")

        return voltage_eval

    def species_lim(self, SV, val):
        """
        Check to see if the minimum species concentration limit has been exceeded.
        """
        # Save local copies of the solution vector and pointers for this electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]

        # Default is that the minimum hasn't been exceeded:
        species_eval = 1.

        # For each electrode point, find the minimum species concentration, and # compare to the user-provided minimum.  Save only the minimum value:
        for j in range(self.n_points):
            Ck_loc = SV_loc[SVptr['C_k_elyte'][j,:]]

            local_eval = min(Ck_loc) - val
            species_eval = min(species_eval, local_eval)

            if np.isnan(np.sum(Ck_loc)):
                species_eval = -1

                break

        # The simulation  looks for instances where this value changes sign
        # (i.e. where it equals zero)
        return  species_eval

    def adjust_separator(self, sep):
        """
        Sometimes, an electrode object requires adjustments to the separator object.  This is not the case, for the SPM.
        """

        # Return the separator class object, unaltered:
        return sep

    def output(self, axs, solution, SV_offset, ax_offset):
        """Plot the intercalation fraction vs. time"""
        for j in np.arange(self.n_points):
            eps_product_ptr = (SV_offset + self.SV_offset
                + self.SVptr['eps_product'][j])
            axs[ax_offset].plot(solution[0,:]/3600,
                solution[eps_product_ptr, :])

        axs[ax_offset].set_ylabel(self.name+' product \n volume fraction')

        for name in self.plot_species:
            species_ptr = self.elyte_obj.species_index(name)
            for j in np.arange(self.n_points):
                C_k_elyte_ptr = (SV_offset + self.SV_offset
                    + self.SVptr['C_k_elyte'][j, species_ptr])
                axs[ax_offset+1].plot(solution[0,:]/3600,
                    1000*solution[C_k_elyte_ptr,:])

        axs[ax_offset+1].legend(self.plot_species)
        axs[ax_offset+1].set_ylabel('Elyte Species Conc. \n (mol m$^{-3}$)')
        return axs

#Official Soundtrack:
    #Belle and Sebastian - The Boy with the Arab Strap
    #Passion Pit - Gossamer
    #CHVRCHES - Every Open Eye
    #Cursive - Happy Hollow
    #Japancakes - If I Could See Dallas
    #Jimmy Eat World - Chase the Light + Invented
    #Lay - Lit
    #George Ezra - Staying at Tamara's
#