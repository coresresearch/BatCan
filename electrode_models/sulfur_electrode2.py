"""
    sulfur_electrode.py

    Class file for air electrode methods
"""

import cantera as ct
from math import tanh, pi, isnan, exp
import numpy as np

class electrode():
    """
    Create an electrode object representing the air electrode.
    """

    def __init__(self, input_file, inputs, sep_inputs, an_inputs,
        electrode_name, params, offset):
        """
        Initialize the model.
        """

        # Import relevant Cantera objects.
        self.elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])
        self.host_obj = ct.Solution(input_file, inputs['host-phase'])
        self.host_surf_obj = ct.Interface(input_file, inputs['surf-phase'],
            [self.elyte_obj, self.host_obj])

        # Create conversion phases and associated electrolyte interfaces:
        self.conversion_phases = []
        self.conversion_obj = []
        self.conversion_surf_obj = []
        self.n_conversion_phases = 0
        self.conversion_phase_min = []
        for ph in inputs["conversion-phases"]:
            self.conversion_phases.append(ph["bulk-name"])
            self.conversion_obj.append(ct.Solution(input_file, ph["bulk-name"]))
            self.conversion_surf_obj.append(ct.Interface(input_file,
                ph["surf-name"],[self.elyte_obj, self.host_obj,
                self.conversion_obj[self.n_conversion_phases]]))
            self.conversion_phase_min.append(ph["min-vol-frac"])
            self.n_conversion_phases += 1

        E0_ca = self.host_surf_obj.delta_standard_gibbs/ct.faraday
        print('E_eq_ca =', -E0_ca)

        self.sw_conv = np.ones((self.n_conversion_phases))

        self.index_conversion_elyte = np.zeros((self.n_conversion_phases))
        for ph_index, ph in enumerate(inputs["conversion-phases"]):
            self.index_conversion_elyte[ph_index] = self.elyte_obj.species_index(ph["elyte-name"])

        # Electrode thickness and inverse thickness:
        self.n_points = inputs['n-points']
        self.dy = inputs['thickness']/self.n_points
        self.dyInv = 1/self.dy
        self.dyinv_n = self.n_points/self.dy*np.ones((self.n_points, 1))

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
        self.index_Li = self.elyte_obj.species_index(inputs['mobile-ion'])

        # Phase volume fractions
        self.eps_host = inputs['eps_host']
        self.eps_elyte_init = 1 - self.eps_host
        self.eps_conversion_init = np.zeros((self.n_conversion_phases))
        for item in inputs["initial-state"]["eps_init"]:
            j = self.conversion_phases.index(item["phase"])
            self.eps_conversion_init[j] = item["value"]

        self.sigma_el =inputs['sigma_el']
        # The following calculations assume spherical particles of a single
        # radius, with no overlap.
        self.r_host = inputs['r_host']
        self.V_host = 4./3. * np.pi * (self.r_host)**3  # Volume of a single carbon / host particle [m3]
        self.A_host = 4. * np.pi * (self.r_host / 2)**2 # Surface area of a single carbon / host particle [m2]
        self.A_init = 2e4 #self.eps_host * self.A_host / self.V_host  # m2 of host-electrolyte interface / m3 of total volume [m-1]
        print('Eps_C =', self.eps_host)
        # For some models, the elyte thickness is different from that of the
        # electrode, so we specify it separately:
        self.dy_elyte = self.dy

        # Inverse double layer capacitance, per unit interfacial area.
        self.C_dl_Inv = 1/inputs['C_dl']

        # Microstructure-based transport scaling factor, based on Bruggeman
        # coefficient of -0.5:
        self.elyte_microstructure = self.eps_elyte_init**1.5 # where would we use this?

        # SV_offset specifies the index of the first SV variable for the
        # electode (zero for anode, n_vars_anode + n_vars_sep for the cathode)
        self.SV_offset = offset

        # Determine the electrode capacity (Ah/m2)
        Conc = (self.conversion_obj[inputs["stored-ion"]["n_phase"]].
                concentrations[0])

        #self.capacity = (Conc*inputs['stored-ion']['charge']*ct.faraday
        #        * (1. - inputs['eps_host']))*inputs['thickness']/3600
        self.capacity = 1675*self.eps_conversion_init[0]*inputs['thickness']*self.conversion_obj[0].density_mass

        # Constant nucleation density of conversion phases
        self.conversion_phase_np = np.zeros([self.n_conversion_phases])
        for item in inputs["nucleation-density"]:
            j = self.conversion_phases.index(item["phase"])
            self.conversion_phase_np[j] = item["value"]

        self.conversion_phase_np[1] = 5e13*exp(2.4221*params['simulations'][0]['C-rate'])

        self.A_conversion_0 = 2*pi*self.conversion_phase_np*(3*self.eps_conversion_init/2/self.conversion_phase_np/pi)**(2/3)
        self.r_conversion_0 = 3*self.eps_conversion_init/self.A_conversion_0
        self.A_C_0 = self.A_init - sum(pi*self.conversion_phase_np*self.r_conversion_0**2)
        print('A_S_0 =', self.A_conversion_0[0])
        print('A_L_0 =', self.A_conversion_0[1])
        print('A_C_0 =', self.A_C_0)
        eps_el_0 = 1 - self.eps_host - sum(self.eps_conversion_init)
        print('porosity =', eps_el_0)
        V_elyte_0 = (inputs['thickness']*eps_el_0
                    +sep_inputs['thickness']*sep_inputs['eps_electrolyte'])
        self.m_S_tot_0 = self.eps_conversion_init[0]*self.conversion_obj[0].density_mass*inputs['thickness']
        E_to_S = 1e3*V_elyte_0/self.m_S_tot_0
        print('Elyte/sulfur ratio =', E_to_S)
        print('Solid sulfur =', self.m_S_tot_0)

        # Number of state variables: electrode potential, double layer
        #   potential, electrolyte composition, oxide volume fraction
        self.n_vars = 2 + self.elyte_obj.n_species + self.n_conversion_phases
        self.n_vars_tot = self.n_points*self.n_vars

        # If user input for non-dimensionalization is true, initialize the
        #   vector that will hold the scaling values for each variable in the SV
        self.scale_nd = np.ones([self.n_vars])
        self.scale_nd_vec = np.tile(self.scale_nd, self.n_points)
        self.nd_type = params['simulations'][0]['non-dimensionalize']
        self.scale_eps_el = 1
        if self.nd_type == 'init' or self.nd_type == 'equil':
            self.scale_nd_flag = 1
        else:
            self.scale_nd_flag = 0

        # Specify the number of plots
        #   1 - Elyte species concentrations for select species
        #   2 - Cathode produce phase volume fraction
        self.n_plots = 1 + 1 #self.n_conversion_phases

        # Store any extra species to be ploted
        self.plot_species = []
        [self.plot_species.append(sp['name']) for sp in inputs['plot-species']]

        self.plot_conv_ph = []
        [self.plot_conv_ph.append(ph['name']) for ph in inputs['plot_conv_ph']]

        # Set Cantera object state:
        self.host_obj.electric_potential = inputs["phi_0"]
        self.host_obj.TP = params["T"], params["P"]
        self.elyte_obj.electric_potential = sep_inputs['phi_0']
        self.elyte_obj.TP = params["T"], params["P"]
        self.host_surf_obj.TP = params["T"], params["P"]
        for ph in self.conversion_obj:
            ph.TP = params["T"], params["P"]

        for ph in self.conversion_surf_obj:
            ph.TP = params["T"], params["P"]

        # Set up pointers to specific variables in the solution vector:
        self.SVptr = {}
        self.SVptr['phi_ed'] = np.empty((self.n_points), dtype=int)
        self.SVptr['phi_dl'] = np.empty((self.n_points), dtype=int)
        self.SVptr['eps_conversion'] = np.empty((self.n_points,
            self.n_conversion_phases), dtype=int)
        self.SVptr['C_k_elyte'] = np.empty((self.n_points,
            self.elyte_obj.n_species), dtype=int)

        for j in np.arange(0, self.n_points):
            self.SVptr['phi_ed'][j] = 0 + j*self.n_vars
            self.SVptr['phi_dl'][j] = 1 + j*self.n_vars

            for ph in np.arange(0, self.n_conversion_phases):
                self.SVptr['eps_conversion'][j, ph] = 2 + ph + j*self.n_vars

            self.SVptr['C_k_elyte'][j, :] = \
                np.arange(self.n_vars*j + 2 + self.n_conversion_phases,
                          self.n_vars*j + 2 + self.n_conversion_phases +
                          self.elyte_obj.n_species, dtype=int)

        self.SVnames = (['phi_ed', 'phi_dl', 'eps_S8', 'eps_Li2S'] + self.elyte_obj.species_names[:])*self.n_points
        # A pointer to where the SV variables for this electrode are, within
        # the overall solution vector for the entire problem:
        self.SVptr['electrode'] = np.arange(offset, offset+self.n_vars_tot)

        # Save the indices of any algebraic variables:
        self.algvars = offset + self.SVptr['phi_ed'][:]

        # Save the indices of constrained variables
        self.constraints_idx = self.SVptr['electrode']
        self.constraints_idx = self.constraints_idx.flatten()
        #self.constraints_type = np.ones_like(self.constraints_idx)
        self.constraints_type = np.zeros([self.n_vars_tot])
        self.constraints_type[self.SVptr['C_k_elyte']] = 1.0
        self.constraints_type[self.SVptr['eps_conversion']] = 1.0

        # Set array of atol to pass to solver
        self.atol = np.ones([self.n_vars_tot])*1e-3
        self.atol[self.SVptr['C_k_elyte']] = 1e-16
        self.atol[self.SVptr['eps_conversion']] = 1e-8

    def initialize(self, inputs, sep_inputs):

        # Initialize the solution vector for the electrode domain:
        SV = np.zeros([self.n_vars_tot])

        # Load intial state variables:
        SV[self.SVptr['phi_ed']] = inputs['phi_0']
        SV[self.SVptr['phi_dl']] = sep_inputs['phi_0'] - inputs['phi_0']
        if inputs["initial-state"]["method"]=="porosity":
            for item in inputs["initial-state"]["eps_init"]:
                j = self.conversion_phases.index(item["phase"])
                SV[self.SVptr['eps_conversion'][:,j]] = item["value"]
        SV[self.SVptr['C_k_elyte']] = self.elyte_obj.concentrations

        if self.scale_nd_flag:
            self.scale_nd = np.copy(SV[0:self.n_vars])
            self.scale_eps_el = 1 - self.eps_host - sum(self.scale_nd[self.SVptr['eps_conversion'][0]])
            self.scale_nd[self.scale_nd == 0] = 1e-12

        self.scale_nd_vec = np.tile(self.scale_nd, self.n_points)

        SV /= self.scale_nd_vec
        return SV

    def residual(self, t, SV, SVdot, sep, counter, params):
        """
        Define the residual for the state of the metal air electrode.

        This is an array of differential and algebraic governing equations, one
        for each state variable in the anode (anode plus a thin layer of electrolyte + separator).

        1. The electric potential in the electrode phase is an algebraic variable.
            In the anode, phi = 0 is the reference potential for the system.
            In the cathode, the electric potential must be such that the ionic
            current is spatially invariant (i.e. it is constant and equal to the
            external applied current, for galvanostatic simulations).

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
            - For either electrode, the sign convention assumes that positive
            flux is from the separator toward the current collector.  All loops
            over the finite volumes proceeds in this same direction.
        """
        # Initialize the residual:
        resid = np.zeros((self.n_vars_tot,))

        # Save local copies of the solution vectors, pointers for this
        #   electrode:
        SVptr = self.SVptr
        scale_nd = self.scale_nd
        scale_nd_vec = self.scale_nd_vec
        #print('cathode scaling', scale_nd_vec)
        SV_loc = SV[SVptr['electrode']]*scale_nd_vec
        SVdot_loc = SVdot[SVptr['electrode']]

        SVdot_loc_dim = SVdot[SVptr['electrode']]*scale_nd_vec

        SV_loc_nd = SV[SVptr['electrode']]
        SVdot_loc_nd = SVdot[SVptr['electrode']]

        # Start at the separator boundary:
        j = self.nodes[0]

        # Read out properties:
        #np_conversion = np.zeros((self.n_conversion_phases))
        #r_conversion = np.zeros((self.n_conversion_phases))
        #A_conversion = np.zeros((self.n_conversion_phases))
        #tpb_conversion = np.zeros((self.n_conversion_phases))
        np_conversion = self.conversion_phase_np
        phi_ed = SV_loc[SVptr['phi_ed'][j]]
        phi_elyte = phi_ed + SV_loc[SVptr['phi_dl'][j]]
        c_k_elyte = SV_loc[SVptr['C_k_elyte'][j]]
        eps_conversion = np.zeros((self.n_conversion_phases))
        for ph_i in range(self.n_conversion_phases):
            eps_conversion[ph_i] = max(SV_loc[SVptr['eps_conversion'][j][ph_i]], 1e-15)
        eps_elyte = 1. - sum(eps_conversion[:]) - self.eps_host
        #if params['i_ext'] != 0:
            #print(eps_conversion)
        #mult = np.zeros((self.n_conversion_phases))

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
            self.elyte_obj.X = c_k_elyte
            for ph in self.conversion_obj:
                ph.electric_potential = phi_ed
            for ph in self.conversion_surf_obj:
                ph.electric_potential = phi_ed

            # Set microstructure multiplier for effective diffusivities
            #TODO #48
            self.elyte_microstructure = eps_elyte**1.5

            # Read out state variables for 'next' node toward CC boundary:
            phi_ed_next = SV_loc[SVptr['phi_ed'][j_next]]
            phi_elyte_next = phi_ed_next + SV_loc[SVptr['phi_dl'][j_next]]
            c_k_elyte_next = SV_loc[SVptr['C_k_elyte'][j_next]]
            eps_conversion_next = SV_loc[SVptr['eps_conversion'][j_next]]
            eps_elyte_next = 1. - sum(eps_conversion_next[:]) - self.eps_host

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

            # Calculate representative particle radius of conversion phases
            r_conversion = (1.5*eps_conversion/np_conversion/pi)**(1/3)

            # Calculate phase/electrolyte surface area for conversion phases
            A_conversion = (3*eps_conversion/r_conversion)

            # Calculate tpb boundary length if applicable
            tpb_conversion = 3*eps_conversion/(r_conversion**2)

            # Calculate available surface area (m2 interface per m3 electrode):
            A_occupied = sum(pi*np_conversion*r_conversion**2)
            A_avail = self.A_init - A_occupied

            # Convert to m2 interface per m2 geometric area:
            A_surf_ratio = A_avail*self.dy
            A_conversion_ratio = A_conversion*self.dy

            # Multiplier to scale phase destruction rates.  As eps_product
            # drops below the user-specified minimum, any reactions that
            # consume the phase have their rates quickly go to zero:
            sw_conv = np.where(eps_conversion < 1e-7, 0, self.sw_conv)
            #sw_conv = np.zeros((self.n_conversion_phases))
            #for ph_index, ph in enumerate(eps_conversion):
            #    if ph < 1e-5 and ph_index == 0:
            #        sw_conv[ph_index] = 0
            #    else:
            #        sw_conv[ph_index] = 1

            # Chemical production rate of the conversion phases:
            # (kmol/m2-interface/s)
            for ph_i, ph in enumerate(self.conversion_surf_obj):
                sdot_conversion = \
                    ph.get_net_production_rates(self.conversion_obj[ph_i])

                nu = self.conversion_obj[ph_i].partial_molar_volumes
                # Rate of change of the conversion phase volume fraction:
                resid[SVptr['eps_conversion'][j][ph_i]] = \
                    (SVdot_loc[SVptr['eps_conversion'][j][ph_i]]
                    - sw_conv[ph_i]*A_conversion[ph_i]
                    * np.dot(sdot_conversion, nu)/scale_nd[SVptr['eps_conversion'][0][ph_i]])

            # Production rate of the electron (moles / m2 interface / s)
            sdot_electron = \
                (self.host_surf_obj.get_net_production_rates(self.host_obj))

            # Positive Faradaic current corresponds to positive charge created
            # in the electrode:
            i_Far = -(ct.faraday * sdot_electron)

            # Double layer current has the same sign as i_Far
            i_dl = self.i_ext_flag*(i_io_in - i_io_out)/A_surf_ratio - i_Far

            resid[SVptr['phi_dl'][j]] = (SVdot_loc[SVptr['phi_dl'][j]] -
                i_dl*self.C_dl_Inv/scale_nd[SVptr['phi_dl'][0]])

            # Species production rate for electrolyte species:
            sdot_elyte_host = \
                self.host_surf_obj.get_net_production_rates(self.elyte_obj)

            R_elyte_conv = np.zeros((self.elyte_obj.n_species,
                                     self.n_conversion_phases))
            for ph_i, ph in enumerate(self.conversion_surf_obj):
                sdot_elyte_conv = ph.get_net_production_rates(self.elyte_obj)
                R_elyte_conv[:,ph_i] = (sdot_elyte_conv) \
                                            *sw_conv[ph_i]*A_conversion[ph_i]

            # The double layer current acts as an additional chemical source/
            # sink term:
            sdot_elyte_host[self.index_Li] -= i_dl / ct.faraday

            R_elyte = np.sum(R_elyte_conv, axis=1) + sdot_elyte_host*A_avail

            # Change in electrolyte species concentration, per unit time:
            dEps_el = -np.dot(scale_nd[SVptr['eps_conversion'][0]], SVdot_loc[SVptr['eps_conversion'][j]])
            theta_eps_el = eps_elyte/self.scale_eps_el
            theta_Ck = SV_loc[SVptr['C_k_elyte'][j]]/scale_nd[SVptr['C_k_elyte'][0]]
            resid[SVptr['C_k_elyte'][j]] = (SVdot_loc[SVptr['C_k_elyte'][j]]
                - (R_elyte + (N_k_in - N_k_out)*self.dyInv)/eps_elyte/scale_nd[SVptr['C_k_elyte'][0]]
                + theta_Ck*dEps_el/eps_elyte)

            # Re-set "next" node state variables to be the new "current" node
            # state variables:
            phi_ed = phi_ed_next
            phi_elyte = phi_elyte_next
            c_k_elyte = c_k_elyte_next
            eps_conversion = eps_conversion_next
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
        self.elyte_obj.X = c_k_elyte
        for ph in self.conversion_obj:
            ph.electric_potential = phi_ed
        for ph in self.conversion_surf_obj:
            ph.electric_potential = phi_ed

        # Electric potential boundary condition:
        if self.name=='anode':
            # The electric potential of the anode = 0 V.
            resid[[SVptr['phi_ed'][j]]] = phi_ed

        elif self.name=='cathode':
            if params['boundary'] == 'current':
                # Total current (i_io + i_el) into the node equals i_ext:
                i_el_out = params['i_ext']
                resid[SVptr['phi_ed'][j]] =  \
                    i_io_in - i_io_out + i_el_in - i_el_out
            elif params['boundary'] == 'potential':
                # Electrode potential = cell potential:
                resid[SVptr['phi_ed'][j]] = (SV_loc[SVptr['phi_ed']]
                    - params['potential'])

        # Calculate representative particle radius of conversion phases
        r_conversion = (1.5*eps_conversion/np_conversion/pi)**(1/3)

        # Calculate phase/electrolyte surface area for conversion phases
        A_conversion = 3*eps_conversion/r_conversion

        # Calculate tpb boundary length if applicable
        tpb_conversion = 3*eps_conversion/(r_conversion**2)

        # Calculate available surface area (m2 interface per m3 electrode):
        A_occupied = sum(pi*np_conversion*r_conversion**2)
        A_avail = self.A_init - A_occupied

        # Convert to m2 interface per m2 geometric area:
        A_surf_ratio = A_avail*self.dy
        A_conversion_ratio = A_conversion*self.dy

        # Multiplier to scale phase destruction rates.  As eps_product drops
        # below the user-specified minimum, any reactions that consume the
        # phase have their rates quickly go to zero:
        sw_conv = np.where(eps_conversion < 1e-5, 0, self.sw_conv)
        #sw_conv = np.zeros((self.n_conversion_phases))
        #for ph_index, ph in enumerate(eps_conversion):
        #    if ph < 1e-5 and ph_index == 0:
        #        sw_conv[ph_index] = 0
        #    else:
        #        sw_conv[ph_index] = 1

        # Chemical production rate of the product phase: (mol/m2 interface/s)
        #for ph in np.arange(0, self.n_conversion_phases):
        #gnpr = [ph.get_net_production_rates for ph in self.conversion_surf_obj]
        #sdot_conversion = [gnpr[ph](self.conversion_obj[ph]) for ph in range(self.n_conversion_phases)]
        #nu = [self.conversion_obj[ph].partial_molar_volumes for ph in range(self.n_conversion_phases)]
        #sdot_nu = [np.dot(sdot_conversion[ph], nu[ph]) for ph in range(self.n_conversion_phases)]
        #resid[SVptr['eps_conversion'][j][:]] = SVdot_loc[SVptr['eps_conversion'][j][:]] - \
        #                            sw_conv[:]*A_conversion[:]*sdot_nu[:]/scale_nd[SVptr['eps_conversion'][0][:]]
        for ph_i, ph in enumerate(self.conversion_surf_obj):
            sdot_conversion = \
                ph.get_net_production_rates(self.conversion_obj[ph_i])

            nu = self.conversion_obj[ph_i].partial_molar_volumes
            # Rate of change of the product phase volume fraction:
            resid[SVptr['eps_conversion'][j][ph_i]] = \
                (SVdot_loc[SVptr['eps_conversion'][j][ph_i]]
                - sw_conv[ph_i]*A_conversion[ph_i]*np.dot(sdot_conversion, nu)/scale_nd[SVptr['eps_conversion'][0][ph_i]])

        # Production rate of the electron (moles / m2 interface / s)
        sdot_electron = \
            (self.host_surf_obj.get_net_production_rates(self.host_obj))

        # Positive Faradaic current corresponds to positive charge created in
        # the electrode (per m2 reaction interface area):
        i_Far = -(ct.faraday * sdot_electron)

        # Double layer current has the same sign as i_Far
        i_dl = self.i_ext_flag*(i_io_in - i_io_out)/A_surf_ratio - i_Far

        resid[SVptr['phi_dl'][j]] = (SVdot_loc[SVptr['phi_dl'][j]]
            - i_dl*self.C_dl_Inv/scale_nd[SVptr['phi_dl'][0]])

        # Molar production rate of electrolyte species at the electrolyte-
        #   electrode interface (kmol / m2 of interface / s)
        sdot_elyte_host = \
            self.host_surf_obj.get_net_production_rates(self.elyte_obj)

        R_elyte_conv = np.zeros((self.elyte_obj.n_species,
                                 self.n_conversion_phases))
        for ph_i, ph in enumerate(self.conversion_surf_obj):
            sdot_elyte_conv = ph.get_net_production_rates(self.elyte_obj)
            R_elyte_conv[:,ph_i] = (sdot_elyte_conv) \
                                        * sw_conv[ph_i]*A_conversion[ph_i]

        # Double layer current represents an additional chemical source/sink
        # term, for electrolyte chemical species:
        sdot_elyte_host[self.index_Li] -= i_dl / ct.faraday

        #sdot_elyte_conv_ph *= A_conversion
        R_elyte = np.sum(R_elyte_conv, axis=1) + sdot_elyte_host*A_avail

        #dEps_el = -sum(SVdot_loc[SVptr['eps_conversion'][j]])
        dEps_el = -np.dot(scale_nd[SVptr['eps_conversion'][0]], SVdot_loc[SVptr['eps_conversion'][j]])
        theta_eps_el = eps_elyte/self.scale_eps_el
        theta_Ck = SV_loc[SVptr['C_k_elyte'][j]]/scale_nd[SVptr['C_k_elyte'][0]]
        # Rate of change of electrolyte chemical species molar concentration:
        resid[SVptr['C_k_elyte'][j]] = (SVdot_loc[SVptr['C_k_elyte'][j]]
            - (R_elyte + (N_k_in - N_k_out)*self.dyInv)/eps_elyte/scale_nd[SVptr['C_k_elyte'][0]]
            + theta_Ck*dEps_el/eps_elyte)  #theta_Ck*dEps_el/theta_eps_el)

        
        return resid

    def voltage_lim(self, SV, val):
        """
        Check to see if the voltage limits have been exceeded.
        """
        # Save local copies of the solution vector and pointers for this electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]*self.scale_nd_vec

        # Calculate the current voltage, relative to the limit.  The simulation
        # looks for instances where this value changes sign (i.e. where it
        # crosses zero)
        voltage_eval = SV_loc[SVptr['phi_ed'][-1]] - val

        return voltage_eval

    def species_lim(self, SV, val):
        """
        Check to see if the min species concentration limit has been exceeded
        """
        # Save local copies of the solution vector and pointers for this electrode
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]*self.scale_nd_vec

        # Default is that the minimum hasn't been exceeded
        species_eval = 1.

        # For each electrode point, find the minimum species concentration, and
        #   compare to the user provided minimum. Save only the minimum value
        for j in range(self.n_points):
            Ck_loc = SV_loc[SVptr['C_k_elyte'][j,:]]

            local_eval = min(Ck_loc) - val
            species_eval = min(species_eval, local_eval)

            if np.isnan(np.sum(Ck_loc)):
                species_eval = -1
                print("nan found")
                break

        return abs(species_eval) + species_eval

    def adjust_separator(self, sep):
        """
        Sometimes, an electrode object requires adjustments to the separator object.  This is not the case, for the SPM.
        """

        # Return the separator class object, unaltered:
        return sep

    def output(self, axs, solution, SV_offset, x_vec, ax_offset):
        """Plot the intercalation fraction vs. time"""
        cap_plot = np.copy(solution[0,:]/3600/self.m_S_tot_0)
        for ph in np.arange(self.n_conversion_phases):
            for j in np.arange(self.n_points):
                eps_product_ptr = (SV_offset + self.SV_offset
                    + self.SVptr['eps_conversion'][j][ph])
                axs[ax_offset].plot(x_vec,
                    solution[eps_product_ptr, :])

            axs[ax_offset].legend(self.plot_conv_ph, loc=1)
            axs[ax_offset].set_ylabel(self.name+' product \n volume fraction')
            axs[ax_offset].set_ylim((0, 0.15))

        for name in self.plot_species:
            species_ptr = self.elyte_obj.species_index(name)
            for j in np.arange(self.n_points):
                C_k_elyte_ptr = (SV_offset + self.SV_offset
                    + self.SVptr['C_k_elyte'][j, species_ptr])
                axs[ax_offset+self.n_conversion_phases-1].plot(x_vec,
                    solution[C_k_elyte_ptr,:])
                axs[ax_offset+self.n_conversion_phases-1].set_ylim((0.0, 2.0))

        axs[ax_offset+self.n_conversion_phases-1].legend(self.plot_species, loc=1)
        axs[ax_offset+self.n_conversion_phases-1].set_ylabel('Elyte Species Conc. \n (kmol m$^{-3}$)')
        axs[ax_offset+self.n_conversion_phases-1].set_ylim((-0.1, 1.1))
        return axs

    def adjust_scale_nd(self, SV):
        # Update the scaling factor after equilibration
        self.scale_nd = np.copy(SV[0:self.n_vars])
        self.scale_nd[self.scale_nd == 0] = 1e-12
        self.scale_nd_vec = np.tile(self.scale_nd, self.n_points)


#Official Soundtrack:
    #Passion Pit - Gossamer
    #CHVRCHES - Every Open Eye
    #Cursive - Happy Hollow
    #Japancakes - If I Could See Dallas
    #Jimmy Eat World - Chase the Light + Invented
    #Lay - Lit
    #George Ezra - Staying at Tamara's
