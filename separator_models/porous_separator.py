"""
    porous_separator.py

    Class file for porous separator methods
"""

import cantera as ct
import numpy as np

from submodels import transport

class separator():
    # Initialize the model.
    def __init__(self, input_file, inputs, params, offset):

        self.elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])

        self.C_k_0 = [species['C_k'] for species in inputs['transport']['diffusion-coefficients']]

        # State variables: electrolyte potential, electrolyte composition (nsp)
        self.n_vars = 1 + self.elyte_obj.n_species

        self.n_points = inputs['n_points']
        self.dy = inputs['thickness']/self.n_points
        self.dyInv = 1 / self.dy
        self.eps_elyte = inputs['eps_electrolyte']

        # Microstructure-based transport scaling factor, based on Bruggeman
        # coefficient of -0.5:
        self.elyte_microstructure = self.eps_elyte**1.5

        self.D_scale_coeff = inputs['transport']['D_scale_coeff']

        self.index_Li = \
            self.elyte_obj.species_index(inputs['transport']['mobile-ion'])

        # Process transport inputs:
        if inputs['transport']['model']=='dilute-solution':
            # Import transport function:

            self.elyte_transport = transport.dilute_solution
            self.D_k = np.zeros_like(self.elyte_obj.X)
            for item in inputs['transport']['diffusion-coefficients']:
                self.D_k[self.elyte_obj.species_index(item['species'])] = \
                    item['D_k']
        else:
            raise ValueError('Please specify a valid electrolyte transport ',
                'model.')

        if inputs['transport'].get('diffusion-scaling') == 'ideal':
            self.scale_diff = transport.scale_diff_ideal
        elif inputs['transport'].get('diffusion-scaling') == 'zhang':
            self.scale_diff = transport.scale_diff_zhang
            self.n_Li_atoms = np.zeros(self.elyte_obj.n_species)
            for i, species in enumerate(self.elyte_obj.species_names):
                self.n_Li_atoms[i] = self.elyte_obj.n_atoms(species, 'Li')

            self.C_Li_0 = np.dot(self.n_Li_atoms, self.C_k_0)
        else:
            print('Warning: No valid diffusion scaling input, using ideal')
            self.scale_diff = transport.scale_diff_ideal

        self.SV_offset = offset

        # Ionic conductivity of bulk electrolyte (S/m):
        self.sigma_io = inputs['sigma_io']

        # This model produces two plots: Electric potential and Li
        # concentration in the separator.
        self.n_plots = 2

        # Set Cantera object state:
        if 'X_0' in inputs:
            self.elyte_obj.TPX = (params['T'], params['P'],
                inputs['X_0'])
        else:
            self.elyte_obj.TP = params['T'], params['P']

        self.elyte_obj.electric_potential = inputs['phi_0']

    def initialize(self, inputs):
        SV = np.zeros([self.n_points*self.n_vars])

        # Set up pointers:
        self.SVptr = {}
        self.SVptr['phi'] = np.arange(0,self.n_points * self.n_vars,
            self.n_vars, dtype='int')

        self.SVptr['C_k_elyte'] = np.ndarray(shape=(self.n_points,
            self.elyte_obj.n_species), dtype='int')
        for i in range(self.n_points):
            self.SVptr['C_k_elyte'][i,:] = range(1 + i*self.n_vars,
                1 + i*self.n_vars + self.elyte_obj.n_species)

        # What portion of the SV represents the separator?
        self.SVptr['sep'] = np.arange(self.SV_offset,
            self.SV_offset + self.n_points*self.n_vars)

        self.SVnames = \
            (['phi_elyte']+ self.elyte_obj.species_names[:])*self.n_points

        # Save indices for any algebraic variables.
        self.algvars = self.SV_offset + self.SVptr['phi']

        # Load intial state variables:
        SV[self.SVptr['phi']] = inputs['phi_0']
        for i in range(self.n_points):
            SV[self.SVptr['C_k_elyte'][i,:]] = self.elyte_obj.concentrations

        return SV

    def residual(self, SV, SVdot, an, ca, params):
        """
        Define the residual for the state of the porous separator.

        This is an array of differential and algebraic governing equations, one for each state variable in the separator.

        1. The electric potentials are algebraic variables, and must be such that the ionic current is spatially invariant (i.e. it is constant and equal to the external applied current, for galvanostatic simulations).

        The residual corresponding to these variables (suppose an index 'j') are of the form:
            resid[j]  = (epression equaling zero; in this case i_io[j] - i_ext)

        2. All other variables are governed by differential equations, i.e. we have a means to calculate dSV[j]/dt for a state variable SV[j] (state variable with index j).  The residuals corresponding to these variables will have the form:
            resid[j] = SVdot[j] - (expression equalling dSV/dt)

        Inputs:
            - SV: the solution vector representing the state of the entire battery domain.
            - SVdot: the time derivative of each state variable: dSV/dt
            - an: the object representing the anode
            - sep: the object representing the separator
            - ca: the object representing the cathode
            - params: dict of battery simulation parameters.
        """

        # Save local copies of the solution vectors, pointers for the separator:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['sep']]
        SVdot_loc = SVdot[SVptr['sep']]

        # Initialize the residual vector, assuming dSVdt = 0 (we will overwrite/
        #  replace this, as necessary)
        resid = SVdot[self.SVptr['sep']]

        # Calculate the electrolyte species fluxes and the corresponding ionic
        # current at the anode boundary:
        N_k_elyte_in, i_io_in = self.electrode_boundary_flux(SV, an,
            params['T'])

        # The ionic current must equal the external current.
        # resid[self.SVptr['phi'][0]] = i_io - params['i_ext']

        # Repeat this for the electric potential in the other separator nodes:
        for j in np.arange(self.n_points-1):
            # Calculate the electrolyte species fluxes and the corresponding
            # ionic current at the boundary between this separator node and the
            # next one toward the cathode:
            N_k_elyte_out, i_io_out = self.elyte_flux(SV_loc, j, params['T'])

            # The ionic current must equal the external current.
            resid[self.SVptr['phi'][j]] = i_io_in - i_io_out# - params['i_ext']

            resid[self.SVptr['C_k_elyte'][j]] = \
                (SVdot_loc[self.SVptr['C_k_elyte'][j]]
                - (N_k_elyte_in - N_k_elyte_out) * self.dyInv / self.eps_elyte)

            N_k_elyte_in, i_io_in = N_k_elyte_out, i_io_out

        j = self.n_points-1
        N_k_elyte_out, i_io_out = self.electrode_boundary_flux(SV, ca,
            params['T'])

        resid[self.SVptr['C_k_elyte'][j]] = \
            (SVdot_loc[self.SVptr['C_k_elyte'][j]]
            - (N_k_elyte_in - N_k_elyte_out) * self.dyInv / self.eps_elyte)

        # The ionic current must equal the external current.
        resid[self.SVptr['phi'][j]] = i_io_in - i_io_out

        return resid

    def electrode_boundary_flux(self, SV, ed, T):
        """
        Calculate the species fluxes and ionic current between a node in the separator and one of the electrodes.

        This will eventually serve mainly as a pass-thru to an aliased transport function.

        Inputs:
        - SV: full battery solution vector
        - ed: relevant electrode object
        - sep: separator object
        - T: local temperature
        """

        # Determine which indices are at the electrode/electrolyte boundary:
        if ed.name=='anode':
            j_ed = -1
            j_elyte = 0
        elif ed.name=='cathode':
            j_ed = 0
            j_elyte = -1

        # Determine the electrolyte properties in the separator and electrode domains. Let the separator be "state_1," the electrode "state_2"

        # Elyte electric potential in separator:
        phi_1 = SV[self.SVptr['sep'][self.SVptr['phi'][j_elyte]]]

        # Elyte electric potential in electrode:
        phi_ed = SV[ed.SVptr['electrode'][ed.SVptr['phi_ed'][j_ed]]]
        phi_dl = SV[ed.SVptr['electrode'][ed.SVptr['phi_dl'][j_ed]]]
        phi_2 = phi_ed + phi_dl

        C_k_1 = SV[self.SVptr['sep'][self.SVptr['C_k_elyte'][j_elyte]]]
        C_k_2 = SV[ed.SVptr['electrode'][ed.SVptr['C_k_elyte'][j_ed]]]

        # Create dictionaries to pass to the transport function:
        state_1 = {'C_k': C_k_1, 'phi':phi_1, 'T':T, 'dy':self.dy,
            'microstructure':self.elyte_microstructure}
        state_2 = {'C_k': C_k_2, 'phi':phi_2, 'T':T, 'dy':ed.dy_elyte,
            'microstructure':ed.elyte_microstructure}

        # Multiply by ed.i_ext_flag: fluxes are out of the anode, into the cathode.
        N_k_elyte, i_io = tuple(x*ed.i_ext_flag
            for x in self.elyte_transport(state_1, state_2, self))

        return N_k_elyte, i_io

    def elyte_flux(self, SV, j, T):
        """
        Calculate the species fluxes and ionic current between two adjacent nodes in the separator.

        This will eventually serve mainly as a pass-thru to an aliased transport function.

        Inputs:
        - SV: separator-relevant portion of the solution vector
        - sep: separator object
        - j: index for relevant location of the separator
        - T: local temperature
        """

        # TEMPORARY: set species fluxes to zero.
        N_k_elyte = np.zeros_like(self.elyte_obj.X)

        # Read out local and adjacent electrolyte properties:
        phi_1 = SV[self.SVptr['phi'][j]]
        phi_2 = SV[self.SVptr['phi'][j+1]]

        C_k_1 = SV[self.SVptr['C_k_elyte'][j]]
        C_k_2 = SV[self.SVptr['C_k_elyte'][j+1]]

        # Create dictionaries to pass to the transport function:
        state_1 = {'C_k': C_k_1, 'phi':phi_1, 'T':T, 'dy':self.dy,
            'microstructure':self.elyte_microstructure}
        state_2 = {'C_k': C_k_2, 'phi':phi_2, 'T':T, 'dy':self.dy,
            'microstructure':self.elyte_microstructure}

        N_k_elyte, i_io = self.elyte_transport(state_1, state_2, self)

        return N_k_elyte, i_io

    def output(self, axs, solution, an, ca, SV_offset, ax_offset):

        phi_elyte_ptr = np.add(self.SV_offset+(self.SVptr['phi']), SV_offset)

        phi_an = (solution[an.SVptr['phi_ed'][0]+SV_offset,:]
            + solution[an.SVptr['phi_dl'][0]+SV_offset,:])
        axs[ax_offset].plot(solution[0,:]/3600, phi_an)
        for j in np.arange(self.n_points):
            axs[ax_offset].plot(solution[0,:]/3600,
                solution[phi_elyte_ptr[j],:])

        phi_ca = \
            (solution[ca.SVptr['electrode'][ca.SVptr['phi_ed'][0]]+SV_offset,:] + solution[ca.SVptr['electrode'][ca.SVptr['phi_dl'][0]]+SV_offset,:])
        axs[ax_offset].plot(solution[0,:]/3600, phi_ca)
        axs[ax_offset].set_ylabel('Separator Potential \n(V)')

        # Axis 5: Li+ concentration:
        Ck_elyte_an = solution[an.SVptr['C_k_elyte'][0]+SV_offset,:]
        axs[ax_offset+1].plot(solution[0,:]/3600,
            Ck_elyte_an[an.index_Li_elyte,:], label="an interface")

        Ck_elyte_sep_ptr = \
            np.add(self.SV_offset+self.SVptr['C_k_elyte'],SV_offset)
        for j in np.arange(self.n_points):
            axs[ax_offset+1].plot(solution[0,:]/3600,
                solution[Ck_elyte_sep_ptr[j, self.index_Li],:],
                label="separator "+str(j+1))

        for j in range(int(ca.n_points)):
            Ck_elyte_ca = \
                solution[ca.SV_offset+ca.SVptr['C_k_elyte'][j]+SV_offset,:]
            axs[ax_offset+1].plot(solution[0,:]/3600,
                Ck_elyte_ca[ca.index_Li_elyte,:])

        axs[ax_offset+1].set_ylabel('Li+ concentration \n(kmol/m$^3$')

        return axs

    def species_lim(self, SV, val):
        """
        Check to see if the minimum species concentration limit has been exceeded.
        """
        # Save local copies of the solution vector and pointers for this electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['sep']]

        # Default is that the minimum hasn't been exceeded:
        species_eval = 1.

        # For each electrode point, find the minimum species concentration, and # compare to the user-provided minimum.  Save only the minimum value:
        for j in range(self.n_points):
            local_eval = min(SV_loc[SVptr['C_k_elyte'][j,:]]) - val
            species_eval = min(species_eval, local_eval)

        # The simulation  looks for instances where this value changes sign
        # (i.e. where it crosses zero)
        return species_eval
