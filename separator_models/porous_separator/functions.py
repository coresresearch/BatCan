"""
    porous_separator.py

    Class file for porous separator methods
"""

import cantera as ct
import numpy as np

def residual(SV, SVdot, an, sep, ca, params):
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
    SVptr = sep.SVptr
    SV_loc = SV[SVptr['sep']]
    SVdot_loc = SVdot[SVptr['sep']]

    # Initialize the residual vector, assuming dSVdt = 0 (we will overwrite/
    #  replace this, as necessary)
    resid = SVdot[sep.SVptr['sep']]
    
    # Calculate the electrolyte species fluxes and the corresponding ionic 
    # current at the anode boundary:
    N_k_elyte, i_io = electrode_boundary_flux(SV, an, sep, params['T'])
    # The ionic current must equal the external current.
    resid[sep.SVptr['phi'][0]] = i_io - params['i_ext']
    
    # Repeat this for the electric potential in the other separator nodes:
    for j in np.arange(sep.n_points-1):
        N_k_elyte_in = N_k_elyte
        # Calculate the electrolyte species fluxes and the corresponding ionic 
        # current at the boundary between this separator node and the next one 
        # toward the cathode:
        N_k_elyte, i_io = elyte_flux(SV_loc, sep, j, params['T'])
        
        # The ionic current must equal the external current.
        resid[sep.SVptr['phi'][j+1]] = i_io - params['i_ext']
        
        resid[sep.SVptr['C_k_elyte'][j]] = \
            (SVdot_loc[sep.SVptr['C_k_elyte'][j]] - (N_k_elyte_in - N_k_elyte)
            * sep.dyInv / sep.eps_elyte)

    j = sep.n_points-1
    N_k_elyte_in = N_k_elyte    
    N_k_elyte, i_io = electrode_boundary_flux(SV, ca, sep, params['T'])

    resid[sep.SVptr['C_k_elyte'][j]] = (SVdot_loc[sep.SVptr['C_k_elyte'][j]] 
        - (N_k_elyte_in - N_k_elyte) * sep.dyInv / sep.eps_elyte) 

    return resid

def elyte_flux(SV, sep, j, T):
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
    N_k_elyte = np.zeros_like(sep.elyte_obj.X)

    # Read out local and adjacent electrolyte properties:
    phi_1 = SV[sep.SVptr['phi'][j]]
    phi_2 = SV[sep.SVptr['phi'][j+1]]

    C_k_1 = SV[sep.SVptr['C_k_elyte'][j]]
    C_k_2 = SV[sep.SVptr['C_k_elyte'][j+1]]

    # Create dictionaries to pass to the transport function:
    state_1 = {'C_k': C_k_1, 'phi':phi_1, 'T':T, 'dy':sep.dy, 
        'microstructure':sep.elyte_microstructure}
    state_2 = {'C_k': C_k_2, 'phi':phi_2, 'T':T, 'dy':sep.dy, 
        'microstructure':sep.elyte_microstructure}

    N_k_elyte, i_io = sep.elyte_transport(state_1, state_2, sep)

    return N_k_elyte, i_io

def electrode_boundary_flux(SV, ed, sep, T):
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
    phi_1 = SV[sep.SVptr['sep'][sep.SVptr['phi'][j_elyte]]]

    # Elyte electric potential in electrode:
    phi_ed = SV[ed.SVptr['electrode'][ed.SVptr['phi_ed'][j_ed]]]
    phi_dl = SV[ed.SVptr['electrode'][ed.SVptr['phi_dl'][j_ed]]]
    phi_2 = phi_ed + phi_dl
    
    C_k_1 = SV[sep.SVptr['sep'][sep.SVptr['C_k_elyte'][j_elyte]]]
    C_k_2 = SV[ed.SVptr['electrode'][ed.SVptr['C_k_elyte'][j_ed]]]

    # Create dictionaries to pass to the transport function:
    state_1 = {'C_k': C_k_1, 'phi':phi_1, 'T':T, 'dy':sep.dy, 
        'microstructure':sep.elyte_microstructure}
    state_2 = {'C_k': C_k_2, 'phi':phi_2, 'T':T, 'dy':ed.dy_elyte, 
        'microstructure':ed.elyte_microstructure}

    # Multiply by ed.i_ext_flag: fluxes are out of the anode, into the cathode.
    N_k_elyte, i_io = tuple(x*ed.i_ext_flag 
        for x in sep.elyte_transport(state_1, state_2, sep))

    return N_k_elyte, i_io


def electrode_boundary_potential(SV, ed, sep):
    """
    Calculate the effective distance between node centers at the electrode/electrolyte boundary and the electric potential in the electrolyte phase on the electrode side of this boundary.

    Inputs:
    - SV: full battery solution vector
    - ed: relevant electrode object
    - sep: separator object
    """
    # Elyte electric potential in anode:
    phi_ed = SV[ed.SVptr['electrode'][ed.SVptr['phi_ed']]]
    phi_dl = SV[ed.SVptr['electrode'][ed.SVptr['phi_dl']]]
    phi_elyte_ed = phi_ed + phi_dl
    
    # Effective distance between node centers, weighted by the electrolyte 
    # microstructure factors:
    dy_elyte_eff = 0.5*(sep.dy/sep.elyte_microstructure 
            + ed.dy/ed.elyte_microstructure)

    return dy_elyte_eff, phi_elyte_ed

def output(axs, solution, an, sep, ca, lp, offset):
    
    phi_elyte_ptr = np.add(sep.SV_offset+(sep.SVptr['phi']), 2)
    
    phi_elyte_an = (solution[an.SVptr['phi_ed'][0]+2,:] 
        + solution[an.SVptr['phi_dl'][0]+2,:])
    axs[offset+1].plot(solution[0,:]/3600, phi_elyte_an)
    for j in np.arange(sep.n_points):
        axs[offset+1].plot(solution[0,:]/3600, solution[phi_elyte_ptr[j],:])

    phi_elyte_ca = (solution[ca.SVptr['electrode'][ca.SVptr['phi_ed'][0]]+2,:] 
        + solution[ca.SVptr['electrode'][ca.SVptr['phi_dl'][0]+2],:])
    axs[offset+1].plot(solution[0,:]/3600, phi_elyte_ca)
    axs[offset+1].set_ylabel('Separator Potential \n(V)',labelpad=lp)
    
    # Axis 5: Li+ concentration:
    Ck_elyte_an = solution[an.SVptr['C_k_elyte'][0]+2,:]
    axs[offset+2].plot(solution[0,:]/3600, Ck_elyte_an[an.index_Li,:],
        label="an interface")

    if 1:
        Ck_elyte_sep_ptr = np.add(sep.SV_offset+sep.SVptr['C_k_elyte'],2)
        for j in np.arange(sep.n_points):
            axs[offset+2].plot(solution[0,:]/3600, 
                solution[Ck_elyte_sep_ptr[j,sep.index_Li],:], 
                label="separator "+str(j+1))

    Ck_elyte_ca = solution[ca.SV_offset+ca.SVptr['C_k_elyte'][0]+2,:]
    axs[offset+2].plot(solution[0,:]/3600, Ck_elyte_ca[ca.index_Li,:])

    axs[offset+2].set_ylabel('Li+ concentration \n(kmol/m$^3$',labelpad=lp)

    return axs
