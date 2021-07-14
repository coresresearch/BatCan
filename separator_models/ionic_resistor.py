"""
    porous_separator.py

    Class file for porous separator methods
"""

import cantera as ct
import numpy as np

class separator():
# Initialize the model.
    def __init__(self, input_file, inputs, params, offset):
        # State variables: electrolyte potential
        self.n_vars = 1
    
        self.dy = inputs['thickness']
        self.dyInv = 1/self.dy
        self.eps_elyte = inputs['eps_electrolyte']

        # Microstructure-based transport scaling factor, based on Bruggeman 
        # coefficient of -0.5:
        self.elyte_microstructure = self.eps_elyte**1.5

        # Ionic conductivity of bulk electrolyte (S/m):
        self.sigma_io = inputs['sigma_io']

        self.SV_offset = offset

        self.n_points = 1

        # This model produces one plot: Electric potential in the separator.
        self.n_plots = 1
    
    def initialize(self, inputs):
        
        # Set up pointers:
        self.SVptr = {}
        self.SVptr['phi'] = np.array([0])
        self.SVptr['sep'] = np.arange(self.SV_offset, 
            self.SV_offset + self.n_vars)

        # Save indices for any algebraic variables.
        self.algvars = [self.SV_offset + self.SVptr['phi'][0]]
        SV = np.zeros([self.n_vars])
    
        # Load intial state variables:
        SV[self.SVptr['phi']] = inputs['phi_0']

        return SV

    def residual(self, SV, SVdot, an, ca, params):
        """
        Define the residual for the state of the separator.

        This is a single algebraic governing equation to determine the separator electric potential.  The separator electric potential must be such that the ionic current is spatially invariant (i.e. it is constant and equal to the external applied current, for galvanostatic simulations).  

        The residual corresponding to this variable (suppose an index 'j') is of the form:
                resid[j]  = (epression equaling zero; here i_io - i_ext)
        """
        # Initialize the residual vector, assuming dSVdt = 0 (we will overwrite/
        #  replace this, as necessary)
        resid = SVdot[self.SVptr['sep']]

        # For the galvanostatic boundary condition, the ionic current must 
        # equal the external current:
        N_k, i_io_an = self.electrode_boundary_flux(SV, an, params['T'])
        
        # Calculate the residual:
        resid[self.SVptr['phi'][0]] = i_io_an - params['i_ext']

        return resid

    def electrode_boundary_flux(self, SV, ed, T):
        """
        Calculate the species fluxes and ionic current between a node in the separator and one of the electrodes.
        """

        # Determine which indices are at the electrode/electrolyte boundary:
        if ed.name=='anode':
            j_ed = -1
        elif ed.name=='cathode':
            j_ed = 0

        # Initialize species fluxes:    
        N_k_elyte = np.zeros_like(ed.elyte_obj.X)

        # Elyte electric potential in electrode:
        phi_ed = SV[ed.SVptr['electrode'][ed.SVptr['phi_ed'][j_ed]]]
        phi_dl = SV[ed.SVptr['electrode'][ed.SVptr['phi_dl'][j_ed]]]
        phi_elyte_ed = phi_ed + phi_dl
        
        # Elyte electric potential in separator:
        phi_elyte_sep = SV[self.SVptr['sep'][self.SVptr['phi'][0]]]
        
        # Average electronic resistance:
        dy_eff = 0.5*(self.dy / self.elyte_microstructure 
                + ed.dy / ed.elyte_microstructure)
        
        # Ionic current:
        i_io = ed.i_ext_flag*(phi_elyte_sep - phi_elyte_ed)*self.sigma_io/dy_eff
        
        # Convert this to flux of the lithium ion:
        N_k_elyte[ed.index_Li] = i_io/ct.faraday/ed.elyte_obj.charges[ed.index_Li]

        return N_k_elyte, i_io

    def electrode_boundary_potential(self, SV, ed):
        """
        Calculate the effective distance between node centers at the electrode/electrolyte boundary and the electric potential in the electrolyte phase on the electrode side of this boundary.
        """
        # Elyte electric potential in electrode:
        phi_ed = SV[ed.SVptr['electrode'][ed.SVptr['phi_ed'][-1]]]
        phi_dl = SV[ed.SVptr['electrode'][ed.SVptr['phi_dl'][-1]]]
        phi_elyte_ed = phi_ed + phi_dl
        
        # Effective distance between node centers, weighted by the electrolyte 
        # microstructure factors:
        dy_elyte_eff = 0.5*(self.dy/self.elyte_microstructure 
                + ed.dy/ed.elyte_microstructure)

        return dy_elyte_eff, phi_elyte_ed

    def output(self, axs, solution, an, ca, ax_offset):
        
        phi_elyte_ptr = np.add(self.SV_offset+(self.SVptr['phi']), 2)
        
        phi_an = (solution[an.SVptr['phi_ed'][0]+2,:] 
            + solution[an.SVptr['phi_dl'][0]+2,:])

        axs[ax_offset].plot(solution[0,:]/3600, phi_an)
        
        for j in np.arange(self.n_points):
            axs[ax_offset].plot(solution[0,:]/3600, 
                solution[phi_elyte_ptr[j],:])

        phi_ca = (solution[ca.SVptr['electrode'][ca.SVptr['phi_ed'][0]]+2,:] 
            + solution[ca.SVptr['electrode'][ca.SVptr['phi_dl'][0]+2],:])
        
        axs[ax_offset].plot(solution[0,:]/3600, phi_ca)
        axs[ax_offset].set_ylabel('Separator Potential \n(V)')
        
        return axs
