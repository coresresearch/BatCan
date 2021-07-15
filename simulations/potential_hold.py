"""
    potential_hold.py

    Functions and outputs for a potentiostatic hold

    Function definitions in this file:
        - 'run' runs the model.
        - 'residual' implements the governing DAE equations to calculate the residual at each time.  This is called by the integrator.
        - 'output' prepares and creates relevant figures and data and saves them to the specified location.

    The methods 'run' and 'output' are called by bat_can.py.  All other functions are called internally.

"""
import numpy as np
from scikits.odes.dae import dae

def run(SV_0, an, sep, ca, algvars, params):
    """ 
    Run the simulation
    """

    # Store the location of all algebraic variables.
    params['algvars'] = algvars

    # Initialize the vector of state vector time derivative (dSV/dt):
    SVdot_0 = np.zeros_like(SV_0)
    
    # This function checks to see if certain limits are exceeded which will 
    # terminate the simulation:
    def terminate_check(t, SV, SVdot, return_val, inputs):
        #TODO: implement concentration check.
        return_val[0] = 1.0
        return_val[1] = 1.0

    # Set up the differential algebraic equation (dae) solver:
    options =  {'user_data':(an, sep, ca, params), 'rtol':1e-8, 'atol':1e-11, 
            'algebraic_vars_idx':algvars, 'first_step_size':1e-18, 
            'rootfn':terminate_check, 'nr_rootfns':2, 'compute_initcond':'yp0'}
    solver = dae('ida', residual, **options)

    # If requested by the user, begin with a hold at zero current, to 
    # equilibrate the system:
    if params['simulation']['equilibrate']['enable']:

        params['boundary'] = 'current'
        params['i_ext'] = 0.0

        print('Step 1: Equilibrating...\n')
        print('    i_ext = 0.0 A/cm2.\n')

        # Read out and set the OCV hold time:
        t_equil = params['simulation']['equilibrate']['time']
        t_out = np.linspace(0, t_equil, 10000)

        # Run the solver
        solution =solver.solve(t_out, SV_0, SVdot_0)

        # Array of current densities, one for each time step taken:
        i_ext = np.zeros_like(solution.values.t)
        
        # Stack the times, current densities, and state vectors:
        data_out = np.vstack((solution.values.t, i_ext, solution.values.y.T))
        
        # Use solution at the end of simulation as the new initial condition:
        SV_0 = solution.values.y[-1,:]

        # Initialize the step counter to 1:
        i_step = 1

    else:
        # Initialize the step counter:
        i_step = 0

    # Specify the boundary condition as potentiostatic:
    params['boundary'] = 'potential'

    for step in params['simulation']['steps']:
        
        # Store the potential:
        params['potential'] = step['potential']

        # Print out the conditions:
        print('Step {:0.0f}: Potentiostatic hold...\n'.format(i_step+1))
        print('    Potential = ', round(params['potential'], 3),' V \n')

        # Determine the range of times to simulate:
        t_out = np.linspace(0, step['time'], 10000)

        # Run the solver
        solution =solver.solve(t_out, SV_0, SVdot_0)

        # Calculate the external current at each time step:
        i_ext = calc_current(solution, sep, an, params)

        # Append the current data array to any preexisting data, for output.  
        # If this is the first step, create the output data array.
        if i_step: # Not the first step. 'data_out' already exists:
            # Stack the times, the current at each time step, and the solution 
            # vector at each time step into a single data array.
            
            # For the times, add the final time from the previous step to all 
            # values:
            SV = np.vstack((solution.values.t+data_out[0,-1], i_ext, 
                solution.values.y.T))
            data_out = np.hstack((data_out, SV))

            # Use SV at the end of the simulation as the new initial condition:
            SV_0 = solution.values.y[-1,:]
        else: # First step. 'data_out' does not yet exist:
            # Stack the times, the current at each time step, and the solution 
            # vector at each time step into a single data array.
            SV = np.vstack((solution.values.t, i_ext, solution.values.y.T))
            data_out = SV

            # Use SV at the end of the simulation as the new initial condition:
            SV_0 = solution.values.y[-1,:]
        
        # Increment the step number:
        i_step += 1

    return data_out

def calc_current(solution, sep, ed, params):
    """
    Calculates the external current from the the state vector.  Since the ionic current in the separator should be equal to the external current at any given point in time, we'll use the current at the anode boundary:
    """

    i_ext = np.zeros_like(solution.values.t)
    for i, SV in enumerate(solution.values.y):
            _, i_ext[i] = sep.electrode_boundary_flux(SV, ed, params['T'])

    return i_ext

def residual(t, SV, SVdot, resid, inputs):
    """
    Call the individual component residual functions, which implement 
    governing equations to solve for the state at any given time step. 
    Nothing is returned by this function. It merely needs to set the value of 
    'resid':
    """
    an, sep, ca, params = inputs

    # Call residual functions for anode, separator, and cathode. Assemble them 
    # into a single residual vector 'resid':
    resid[an.SVptr['electrode']] = an.residual(SV, SVdot, sep, ca, params)

    resid[sep.SVptr['sep']] = sep.residual(SV, SVdot, an, ca, params)
    
    resid[ca.SVptr['electrode']] = ca.residual(SV, SVdot, sep, an, params)

def output(solution, an, sep, ca, params):
    """
    Prepare and save any output data to the correct location. Prepare, 
    create, and save any figures relevant to constant-current cycling.
    """
    #TODO #17
    import matplotlib.pyplot as plt 
    from matplotlib.ticker import FormatStrFormatter

    # Temporary flag for Li metal anode:
    i_Li = 0
    
    # Create figure:
    lp = 30 #labelpad
    # Number of subplots 
    # (this simulation produces 2: current and voltage, vs. time):
    n_plots = 2 + an.n_plots + ca.n_plots + sep.n_plots

    # Initialize the figure:
    fig, axs = plt.subplots(n_plots, 1, sharex=True, 
            gridspec_kw = {'wspace':0, 'hspace':0})
    
    fig.set_size_inches((4.0,1.8*n_plots))
    # Calculate cell potential:   
    phi_ptr = 2+ca.SV_offset+int(ca.SVptr['phi_ed'][:])
 
    # Axis 1: Current vs. capacity
    axs[0].semilogy(solution[0,:]/3600, abs(1000*solution[1,:]/10000))
    axs[0].set_ylabel('abs(Current Density) \n (mA/cm$^2$)',labelpad=lp)
    
    # Axis 2: Charge/discharge potential vs. capacity.
    axs[1].plot(solution[0,:]/3600, solution[phi_ptr,:])
    axs[1].set_ylabel('Cell Potential \n(V)')#,labelpad=lp)

    # Add any relevant anode, cathode, and separator plots: 
    axs = an.output(axs, solution, ax_offset=2)
    axs = ca.output(axs, solution, ax_offset=2+an.n_plots)
    axs = sep.output(axs, solution, an, ca, ax_offset=2+an.n_plots+ca.n_plots)

    axs[n_plots-1].set(xlabel='Time (h)')

    # Format axis ticks:
    for i in range(n_plots):
        axs[i].tick_params(axis="x",direction="in")
        axs[i].tick_params(axis="y",direction="in")
        if i: # Skip the first plot, which is log scale:
            axs[i].get_yaxis().get_major_formatter().set_useOffset(False)
        axs[i].yaxis.set_label_coords(-0.2, 0.5)

    # Round voltage values:
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Trim down whitespace:
    fig.tight_layout()
    
    # Save figure:
    plt.savefig('output.pdf')
    if params['outputs']['show-plots']:
        plt.show()
