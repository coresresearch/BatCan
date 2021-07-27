"""
    cyclic_voltammetry.py

    Functions and outputs for a cyclic voltammetry (CV) experiment

    Function definitions in this file:
        - 'run' runs the model.
        - 'setup_cycles' reads out user inputs and creates arrays of potentials and correponding times to describe the cell potential as a funciton of time during the CV.
        - 'read_sweep_rate' reads the user-supplied sweep rate and converts the units, as necessary, to V/s.
        - 'residual' implements the governing DAE equations to calculate the residual at each time.  This is called by the integrator.
        - 'calc_current' calculates the resulting current density produced by the battery cell at each time step.
        - 'data_prepare' concatenate the simulation results (time steps, external currents, and state vectors) into a single data stack, and appends this to existing data from any previous steps.
        - 'output' prepares and creates relevant figures and data and saves them to the specified location.
        - 'sort_cycles' analyzes the output data and assigns each time step to its relevant cycle number.

    The methods 'run' and 'output' are called by bat_can.py.  All other functions are called internally.

"""
from math import ceil
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
        #TODO #35
        #TODO #34
        return_val[0] = 1.0
        return_val[1] = 1.0

    # Set up the differential algebraic equation (dae) solver:
    options =  {'user_data':(an, sep, ca, params), 'rtol':1e-8, 'atol':1e-11, 
            'algebraic_vars_idx':algvars, 'first_step_size':1e-18, 
            'rootfn':terminate_check, 'nr_rootfns':2, 'compute_initcond':'yp0'}
    solver = dae('ida', residual, **options)

    # Initialize the step counter:
    i_step = 0
    
    """ Equilibration """
    # If requested by the user, begin with a hold at zero current, to 
    # equilibrate the system:
    if params['simulation']['equilibrate']['enable']:

        params['boundary'] = 'current'
        params['i_ext'] = 0.0

        print('\n Equilibrating at i_ext = 0.0 A/cm2.\n')

        # Set the OCV hold time:
        t_equil = params['simulation']['equilibrate']['time']
        t_out = np.linspace(0, t_equil, 10000)

        # Run the solver
        solution = solver.solve(t_out, SV_0, SVdot_0)

        # Array of current densities (all equal to zero), one for each time 
        # step taken:
        i_ext = np.zeros_like(solution.values.t)
        
        # Stack the times, current densities, and state vectors into a single 
        # data stack:
        data_out = np.vstack((solution.values.t, i_ext, solution.values.y.T))
        
        # Use solution at the end of simulation as the new initial condition:
        SV_0 = solution.values.y[-1,:]

        # Increment the step counter:
        i_step += 1

        # Save the potential as the (possible) initial cell potential (it may 
        # be overwritten, below):
        params['simulation']['ocv'] = \
            SV_0[ca.SV_offset+int(ca.SVptr['phi_ed'][-1])]
        
    # Specify the boundary condition as potentiostatic:
    params['boundary'] = 'potential'

    """ Initial Potentiostatic Hold """
    # If requested by the user, begin, with a hold at a specified potential
    if params['simulation']['initial-hold']['enable']:

        # Set the potential for the hold:
        params['potentials'] = \
            np.array((params['simulation']['initial-potential'],
            params['simulation']['initial-potential']))

        # Read out the time for the hold
        t_hold = params['simulation']['initial-hold']['time']

        # Set the times in the input parameters, used to read out the potential 
        # as a function of time:
        params['times'] = np.array((0, t_hold))

        # Set the solver time:
        t_out = np.linspace(0, t_hold, 10000)
        
        # Print out the conditions:
        print('Potentiostatic hold at', round(params['potentials'][0], 3),
            ' V \n')

        # Run the solver
        solution = solver.solve(t_out, SV_0, SVdot_0)

        # Calculate the external current at each time step:
        i_ext = calc_current(solution, sep, an, params)

        # Stack times, current densities, and state vectors into the data array 
        # for output/saving. Save the final state as the new initial state:
        if i_step: # data_out already exists:
            data_out, SV_0 = data_prepare(i_step, solution, i_ext, data_out)
        else: # data_out does not yet exist:
            data_out, SV_0 = data_prepare(i_step, solution, i_ext)
        
    """ Run the CV experiment:"""
    # Set up the array of times and potentials for the CV:
    potentials, times = setup_cycles(params['simulation'])

    # Save the array of potentials and times to the 'params' input:
    params['potentials'] = potentials
    params['times'] = times

    # Tell the solver the range of times to simulate:
    t_out = np.linspace(0, times[-1],1000)
    
    # Run the solver
    solution =solver.solve(t_out, SV_0, SVdot_0)

    # Calculate the external current at each time step:
    i_ext = calc_current(solution, sep, an, params)

    # Append the current data array to any preexisting data, for output.  
    # If this is the first step, create the output data array.
    if i_step: # data_out already exists:
        data_out, SV_0 = data_prepare(i_step, solution, i_ext, data_out)
    else: # data_out does not yet exist:
        data_out, SV_0 = data_prepare(i_step, solution, i_ext)

    return data_out

def setup_cycles(params):
    """
    Creates two arrays, one of cell potentials (V) and one of the corresponding times (s) for each potentia, to describe the CV sweep.  These arrays are interpolated by the residual function.  The points should include the initial potential, the final potential, and any points where the potential hits one of the upper and lower limits."""

    # Read out upper and lower voltage bounds
    phi_bounds = np.array((params['lower-cutoff'], params['upper-cutoff']))

    # Read out the sweep rate, and convert units as necessary to (V/s)
    R = read_sweep_rate(params['sweep-rate'])

    # Time for one complete sweep between the upper and lower bounds:
    dt_sweep = (phi_bounds[1] - phi_bounds[0])/R

    # Direction of the initial sweep: positive-going or negative-going?
    if params['initial-sweep'] == 'positive':
        direction = 1
    elif params['initial-sweep'] == 'negative':
        direction = -1

    # Initial potential:
    if params['initial-potential'] == 'ocv':
        phi_0 = params['ocv']
    else:
        phi_0 = params["initial-potential"]

    # Find the first time where the potential hits one of the limits:
    t_limit_0 = -direction*(phi_0 - phi_bounds[int(0.5*(1. + direction))])/R

    # Make an array containing all of the times when the voltage limits are hit:
    t_events = np.arange(t_limit_0, 
        t_limit_0 + dt_sweep*(2.*params["n_cycles"])+1, 
        dt_sweep)
    
    # Calculate the CV end time (s): 
    t_final = t_limit_0 + dt_sweep*(2.*params["n_cycles"])

    # Concatenate all times into a single array:
    times = np.concatenate((np.array([0.]), t_events, np.array([t_final]),))
    
    # Initialize the array of potentials:
    potentials = np.zeros_like(times)

    # Load the initial potential:
    potentials[0] = phi_0

    # Use the sweep rate, sweep direction, and the `times` array to determine 
    # the other potentials.  Each time a voltage limit is hit, the sweep 
    # changes direction.
    for i, t in enumerate(times[1:]):
        potentials[i+1] = potentials[i] + direction*(t - times[i])*R
        direction *= -1

    return potentials, times

def read_sweep_rate(input):
    """ 
    Reads out the user-supplied sweep rate and converts the units, as necessary, to V/s.
    """
    # The input provides both the sweep rate and its units:
    R, units = input.split()

    # Convert R to a float:
    R = float(R)
    
    # Read the units and convert to V/s as necessary:
    V_units, t_units = units.split("/")

    # Potential units:
    if V_units=="V":
        pass
    elif V_units=="mV":
        R *= 0.001
    elif V_units=="uV":
        R *= 1e-6
    else:
        raise Exception("Please supply sweep rate potential units as V, mV," 
        + " or uV.")

    # Time units:
    if t_units=="s":
        pass
    elif t_units=="min":
        R /= 60
    elif t_units=="ms":
        R *= 1000
    elif t_units=="us":
        R *= 1e6
    else:
        raise Exception("Please supply sweep rate time units as min, s, "
        + "ms, or us.")

    return R

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

def calc_current(solution, sep, ed, params):
    """
    Calculates the external current from the the state vector.  Since the ionic current in the separator should be equal to the external current at any given point in time, we'll use the current at the anode boundary:
    """

    i_ext = np.zeros_like(solution.values.t)
    for i, SV in enumerate(solution.values.y):
            _, i_ext[i] = sep.electrode_boundary_flux(SV, ed, params['T'])

    return i_ext

def data_prepare(i_step, solution, i_ext, data_out=None):
    """
    Concatenate the simulation results (time steps, external currents, and state vectors) into a single data stack, and appends this to existing data from any previous steps.
    """
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

    return data_out, SV_0

def output(solution, an, sep, ca, params):
    """
    Prepare and save any output data to the correct location. Prepare, 
    create, and save any figures relevant to constant-current cycling.
    """
    #TODO #17
    import matplotlib.pyplot as plt 
    from matplotlib.ticker import FormatStrFormatter

    # Re-read the time steps for the CV cycles: points where the potential hits 
    # voltage limits.  The function also retrns the potentials, but these are 
    # not required, here.
    _, times = setup_cycles(params['simulation'])

    # Find the step numbers where each cycle begins:
    indices, n_cycles = sort_cycles(solution, times)

    # Calculate cell potential:   
    phi_ptr = 2+ca.SV_offset+int(ca.SVptr['phi_ed'][:])

    # Plot CV curves:
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches((4.0, 2.25))
    for j in np.arange(n_cycles):
        ax.plot(solution[phi_ptr,indices[j]:indices[j+1]], 
            -0.1*solution[1,indices[j]:indices[j+1]],
            label='cycle '+str(j+1))

    ax.set_ylabel('Current Density (mA/cm$^2$)')
    ax.set_xlabel('Cell Potential (V)')
    ax.legend(frameon=False)

    # Format axis ticks:
    ax.tick_params(axis="x",direction="in")
    ax.tick_params(axis="y",direction="in")
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.yaxis.set_label_coords(-0.2, 0.5)

    # Round voltage values:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Trim down whitespace:
    fig.tight_layout()

    # Save figure:
    plt.savefig('cyclic_voltammogram.pdf')


    # Create time-series subplots figure:
    lp = 30 #labelpad
    # Number of subplots 
    # (this simulation produces 2: current and voltage, vs. time):
    n_plots = 2 + an.n_plots + ca.n_plots + sep.n_plots

    # Initialize the figure:
    fig, axs = plt.subplots(n_plots, 1, sharex=True, 
            gridspec_kw = {'wspace':0, 'hspace':0})
    
    fig.set_size_inches((4.0,1.8*n_plots))
 
    # Axis 1: Current vs. capacity
    axs[0].plot(solution[0,:]/3600, (1000*solution[1,:]/10000))
    axs[0].set_ylabel('Current Density \n (mA/cm$^2$)',labelpad=lp)
    
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

def sort_cycles(solution, times):
    """
    Finds the step number (index) for the beginning of each CV cycle.
    """
    n_cycles = ceil(len(times)/2)
    indices = np.zeros((n_cycles+1,), dtype=int)

    for i, t in enumerate(times[1::2]):
        for j, time in enumerate(solution[0,:]):
            if time >= t:
                indices[i+1] = j
                break

    return indices, n_cycles
