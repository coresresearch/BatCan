"""
    galvanostatic_hold.py

    Functions and outputs for a galvanostatic hold

    Function definitions in this file:
        - 'run' runs the model.
        - 'residual' implements the governing DAE equations to calculate the residual at each time.  This is called by the integrator.
        - 'output' prepares and creates relevant figures and data and saves them to the specified location.

    The methods 'run' and 'output' are called by bat_can.py.  All other functions are called internally.

"""
import numpy as np
from scikits.odes.dae import dae

def run(SV_0, an, sep, ca, algvars, params, sim):
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
        #TODO #36
        return_val[0] = 1.0
        return_val[1] = 1.0

    # Set up the differential algebraic equation (dae) solver:
    options =  {'user_data':(an, sep, ca, params), 'rtol':1e-8, 'atol':1e-11, 
            'algebraic_vars_idx':algvars, 'first_step_size':1e-18, 
            'rootfn':terminate_check, 'nr_rootfns':2, 'compute_initcond':'yp0'}
    solver = dae('ida', residual, **options)

    """ Equilibration: Optional """
    # If requested by the user, begin with a hold at zero current, to 
    # equilibrate the system:
    if sim['equilibrate']['enable']:

        # Set the boundary condition to galvanostatic and the external current 
        # to 0.0:
        params['boundary'] = 'current'
        params['i_ext'] = 0.0

        # Print out conditions:
        print('\nStep 1: Equilibrating...\n')
        print('    i_ext = 0.0 mA/cm2.\n')

        # Read out and set the OCV hold time:
        t_equil = sim['equilibrate']['time']
        t_out = np.linspace(0, t_equil, 10000)

        # Run the solver
        solution = solver.solve(t_out, SV_0, SVdot_0)

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
    params['boundary'] = 'current'

    """ Run the galvanostatic hold(s) """
    for step in sim['steps']:

        # Read the current and units, and convert the units, if necessary. 
           
        # Read in the user input:
        current = step['i_ext']
        # Split the current from the units:
        i_ext, units = current.split()
        # Convert i_ext to a float:
        i_ext = float(i_ext)

        # Read the units and convert i_ext to A/m2 as necessary:
        i_units, A_units = units.split('/')
        if i_units=="mA":
            i_ext *= 0.001
        elif i_units=="uA":
            i_ext *= 1e-6
        if A_units=="cm2":
            i_ext *= 10000

        # Store the current. We do this as an array of current vs. time, 
        # which the residual function interpolates.  For a constant current, 
        # we just need the initial and final currents, which are the same:
        params['i_ext'] = i_ext
        params['times'] = np.array((0, step['time']))

        # Print out the conditions:
        print('Step {:0.0f}: Galvanostatic hold...\n'.format(i_step+1))
        print('    Current density = ', round(i_ext, 3), ' A/m2 \n')

        # Determine the range of times to simulate:
        t_out = np.linspace(0, step['time'], 10000)

        # Run the solver
        solution =solver.solve(t_out, SV_0, SVdot_0)

        # Calculate the external current at each time step:
        current = i_ext*np.ones_like(solution.values.t)

        # Append the current data array to any preexisting data, for output.  
        # If this is the first step, create the output data array.
        if i_step: # Not the first step. 'data_out' already exists:
            # Stack the times, the current at each time step, and the solution 
            # vector at each time step into a single data array.
            
            # For the times, add the final time from the previous step to all 
            # values:
            SV = np.vstack((solution.values.t+data_out[0,-1], current, 
                solution.values.y.T))
            data_out = np.hstack((data_out, SV))

            # Use SV at the end of the simulation as the new initial condition:
            SV_0 = solution.values.y[-1,:]
        else: # First step. 'data_out' does not yet exist:
            # Stack the times, the current at each time step, and the solution 
            # vector at each time step into a single data array.
            SV = np.vstack((solution.values.t, current, solution.values.y.T))
            data_out = SV

            # Use SV at the end of the simulation as the new initial condition:
            SV_0 = solution.values.y[-1,:]
        
        # Increment the step number:
        i_step += 1

    return data_out

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
    resid[an.SVptr['electrode']] = an.residual(t, SV, SVdot, sep, ca, params)

    resid[sep.SVptr['sep']] = sep.residual(SV, SVdot, an, ca, params)
    
    resid[ca.SVptr['electrode']] = ca.residual(t, SV, SVdot, sep, an, params)

def output(solution, an, sep, ca, params, sim, plot_flag=True, 
    return_flag=False, save_flag=True):
    """
    Prepare and save any output data to the correct location. Prepare, 
    create, and save any figures relevant to constant-current cycling.
    """
    #TODO #17
    import matplotlib.pyplot as plt 
    from matplotlib.ticker import FormatStrFormatter
    import os
    import pandas as pd

    # Temporary flag for Li metal anode:
    i_Li = 0
    
    # Create figure:
    lp = 30 #labelpad
    # Number of subplots 
    # (this simulation produces 2: current and voltage, vs. time):
    n_plots = 2 + an.n_plots + ca.n_plots + sep.n_plots

    # There are 2 variables stored before the state variables: (1) time (s), 
    # (2) current density(A/cm2) 
    SV_offset = 2

    # Calculate cell potential:   
    phi_ptr = 2 + ca.SV_offset + int(ca.SVptr['phi_ed'][-1])

    # Save the solution as a Pandas dataframe:
    labels = (['current'] + an.SVnames + sep.SVnames 
        + ca.SVnames)
    solution_df = pd.DataFrame(data = solution.T[:,1:],
                                index = solution.T[:,0],
                                columns = labels)

    solution_df.index.name = 'time (s)'

    # If requested, create output figures:
    if plot_flag:

        # Initialize the figure:
        summary_fig, summary_axs = plt.subplots(n_plots, 1, sharex=True, 
                gridspec_kw = {'wspace':0, 'hspace':0})
        
        summary_fig.set_size_inches((4.0,1.8*n_plots))

        # Axis 1: Current vs. capacity
        summary_axs[0].plot(solution[0,:]/3600, 
            abs(1000*solution[1,:]/10000))
        summary_axs[0].set_ylabel('abs(Current Density) \n (mA/cm$^2$)',
            labelpad=lp)
        
        # Axis 2: Charge/discharge potential vs. capacity.
        summary_axs[1].plot(solution[0,:]/3600, solution[phi_ptr,:])
        summary_axs[1].set_ylabel('Cell Potential \n(V)')#,labelpad=lp)

        # Add any relevant anode, cathode, and separator plots: 
        summary_axs = an.output(summary_axs, solution, SV_offset, ax_offset=2)
        summary_axs = ca.output(summary_axs, solution, SV_offset, 
            ax_offset=2+an.n_plots)
        summary_axs = sep.output(summary_axs, solution, an, ca, SV_offset, 
            ax_offset=2+an.n_plots+ca.n_plots)

        summary_axs[n_plots-1].set(xlabel='Time (h)')

        # Format axis ticks:
        for i in range(n_plots):
            summary_axs[i].tick_params(axis="x",direction="in")
            summary_axs[i].tick_params(axis="y",direction="in")
            summary_axs[i].get_yaxis().get_major_formatter().set_useOffset(False)
            summary_axs[i].yaxis.set_label_coords(-0.2, 0.5)

        # Round voltage values:
        summary_axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Trim down whitespace:
        summary_fig.tight_layout()
    
    # If no specification is given on whether to show plots, assume 'True'    
    if save_flag:
        if 'outputs' not in sim:
            summary_fig.savefig('output.pdf')
            plt.show()
        else:
            if 'save-name' in sim['outputs']:
                if len(params['simulations']) == 1:
                    sim['filename'] = (params['output'] +'_' 
                        + sim['outputs']['save-name'] )
                else:
                    sim['filename'] = (params['output'] +'/'
                        + sim['outputs']['save-name'] )
                
                if not os.path.exists(sim['filename']):
                    os.makedirs( sim['filename'])

                solution_df.to_pickle(sim['filename']+'/output_' 
                    + sim['outputs']['save-name'] + '.pkl')
                solution_df.to_csv(sim['filename']+'/output_' 
                    + sim['outputs']['save-name'] + '.csv', sep=',')
                summary_fig.savefig(sim['filename']+'/summary_' 
                    + sim['outputs']['save-name'] + '.pdf')
            
            if ('show-plots' not in sim['outputs'] or 
                sim['outputs']['show-plots']):
                plt.show()

    if return_flag:
        return solution_df

    # # Save figure:
    # plt.savefig('output.pdf')
    # if sim['outputs']['show-plots']:
    #     plt.show()
