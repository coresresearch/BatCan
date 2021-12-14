"""
    CC_cycle.py

    Functions and outputs for constant-current cycling

    Function definitions in this file:
        - 'run' runs the model.
        - 'calc_current' calculates the external current density (A/m2) from user inputs.
        - 'setup_cycles' Creates two python tuples. One lists the sequence of charge and discharge steps, and the other lists the current density for each step.
        - 'residual' implements the governing DAE equations to calculate the residual at each time.  This is called by the integrator.
        - 'output' prepares and creates relevant figures and data and saves them to the specified location.

    The methods 'run' and 'ouput' are called by bat_can.py.  All other functions are called internally.

"""
import numpy as np
from scikits.odes.dae import dae
from math import floor

def run(SV_0, an, sep, ca, algvars, params, sim):
    """ 
    Run the simulation
    """
    # Determine the current to run at, and the time to fully charge/discharge. 
    # 'calc_current' is defined below.
    current, t_final = calc_current(sim, an, ca)

    # Store the location of all algebraic variables.
    params['algvars'] = algvars

    # Specify the boundary condition as galvanostatic:
    params['boundary'] = 'current'

    # Figure out which steps and at what currents to run the model. This 
    # returns a tuple of 'charge' and 'discharge' steps, and a tuple with a 
    # current for each step.
    steps, currents, times = setup_cycles(sim, current, t_final)

    # This function checks to see if certain limits are exceeded which will 
    # terminate the simulation:
    def terminate_check(t, SV, SVdot, return_val, inputs):
        return_val[0] = ca.voltage_lim(SV, sim['phi-cutoff-lower'])
        return_val[1] = ca.voltage_lim(SV, sim['phi-cutoff-upper'])

    # Set up the differential algebraic equation (dae) solver:
    options =  {'user_data':(an, sep, ca, params), 'rtol':1e-4, 'atol':1e-6, 
            'algebraic_vars_idx':algvars, 'first_step_size':1e-18, 
            'rootfn':terminate_check, 'nr_rootfns':2, 'compute_initcond':'yp0'}
    solver = dae('ida', residual, **options)

    # Go through the current steps and integrate for each current:
    for i, step in enumerate(steps):
        print('Step ',int(i+1),': ',step,'...\n')

        # Set the external current density (A/m2)
        params['i_ext'] = currents[i]
        print('    Current = ', round(currents[i],3),'A/m^2 \n')
        
        t_out = np.linspace(0, times[i], 10000)
        
        # Create an initial array of time derivatives and runs the integrator:
        SVdot_0 = np.zeros_like(SV_0)
        solution = solver.solve(t_out, SV_0, SVdot_0)

        # Create an array of currents, one for each time step:
        i_data = currents[i]*np.ones_like(solution.values.t)

        # Append the current data array to any preexisting data, for output.  
        # If this is the first step, create the output data array.
        if i: # Not the first step. 'data_out' already exists:
            # Stack the times, the current at each time step, and the solution 
            # vector at each time step into a single data array.
            SV = np.vstack((solution.values.t+data_out[0,-1], i_data, solution.
                values.y.T))
            data_out = np.hstack((data_out, SV))

            # Use SV at the end of the simualtion as the new initial condition:
            SV_0 = solution.values.y[-1,:]
        else: # First step. 'data_out' does not yet exist:
            # Stack the times, the current at each time step, and the solution 
            # vector at each time step into a single data array.
            SV = np.vstack((solution.values.t, i_data, solution.values.y.T))
            data_out = SV

            # Use SV at the end of the simualtion as the new initial condition:
            SV_0 = solution.values.y[-1,:]

    return data_out

def calc_current(params, an, ca):
    """
    Calculates the external current from the user inputs.  If a C-rate is given, calculate the battery capacity and convert this to a current.  If 
    i_ext is given, convert the units to A/m2.
    """

    # Battery capacity is the lesser of the anode and cathode capacities. It is 
    # required for determining the simulation time.
    cap = min(an.capacity, ca.capacity)

    if params['i_ext'] is not None: # User specified a current density.
        # User cannot set both i_ext and C-rate. Throw an error, if they have:
        if params['C-rate'] is not None:
            raise ValueError("Both i_ext and C-rate are specified. "
                "Please specify only one of the two in your input file.")
        else:
            # Read the current and units, and convert the units, if necessary. 
            # Read in the user input:
            current = params['i_ext']
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
            
    elif params['C-rate'] is not None: # User specified a C-rate, but not i_ext:
        i_ext = cap*params['C-rate']

    else:
        # If neither i_ext or C_rate is provided, throw an error:
        raise ValueError("Please specify either the external current (i_ext) "
            "or the C-rate (C-rate).")
    
    # Use the capacity divided by current to find the max charge/discharge time:
    t_final = cap*3600/i_ext

    return i_ext, t_final

def setup_cycles(params, current, time):
    """
    Set up a tuple representing steps in the requested charge-discharge cycles.
    Also create a tuple of currents, one for each step.
    """
    steps = ()
    currents = ()
    times = ()

    if params['first-step'] == "discharge":
        cycle = ('discharge', 'charge')
        cycle_currents = (current, -current)
        cycle_times = (time, time)
    else:
        cycle = ('charge','discharge')
        cycle_currents = (-current, current)
        cycle_times = (time, time)
    
    # At present, the only partial cycle accepted is for a single half-cycle 
    # (i.e. a single charge or discharge step).
    #TODO #16
    # For readability:
    n_cycles = params['n_cycles']
    steps = floor(n_cycles)*cycle
    currents = floor(n_cycles)*cycle_currents
    times = floor(n_cycles)*cycle_times
    # Is there a partial cycle at the end?
    partial = n_cycles - floor(n_cycles)
    if partial>0 and partial<=0.5:
        steps = steps + (cycle[0],)
        currents = currents + (cycle_currents[0],)
        times = times + (time * partial * 2.,)
    elif partial > 0.5:
        steps = steps + cycle
        currents = currents + cycle_currents
        times = times + (time, time * (partial - 0.5) * 2.)

    # If requested, start with a hold at open circuit:
    if params['equilibrate']['enable']:
        steps = ('equilibrate',)+ steps
        currents = (0,) + currents
        times = (params['equilibrate']['time'],) + times
        
    return steps, currents, times

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

def output(solution, an, sep, ca, params, sim):
    """
    Prepare and save any output data to the correct location. Prepare, 
    create, and save any figures relevant to constant-current cycling.
    """
    #TODO #17
    import matplotlib.pyplot as plt 
    from datetime import datetime   
    import os 

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
    phi_ptr = 2+ca.SV_offset+int(ca.SVptr['phi_ed'][-1])
 
    # Axis 1: Current vs. capacity
    axs[0].plot(solution[0,:]/3600, 1000*solution[1,:]/10000)
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
        axs[i].get_yaxis().get_major_formatter().set_useOffset(False)
        axs[i].yaxis.set_label_coords(-0.2, 0.5)

    # Trim down whitespace:
    fig.tight_layout()
    
    # Save figure:

    # If no specification is given on whether to show plots, assume 'True'
    if 'outputs' not in sim:
        plt.savefig('output.pdf')
        plt.show()
    else:
        if 'show-plots' not in sim['outputs']:
            plt.show()
        elif sim['outputs']['show-plots']:
            plt.show()

        now = datetime.now()
        dt =  now.strftime("%Y%m%d_%H%M")
        if sim['outputs']['savename']:
            filename = ('outputs/'+sim['outputs']['savename']+'_'+params['input']+'_'+dt)
            os.makedirs(filename)
            print(filename)
            np.savetxt(filename+'/output.csv', solution, delimiter=',')
            plt.savefig(filename+'/summary.pdf')
    
    
