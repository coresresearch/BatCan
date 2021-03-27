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
import cantera as ct
from scikits.odes.dae import dae

def run(SV_0, an, sep, ca, algvars, params):
    # Determine the current to run at. 'calc_current' is defined below.
    current, t_final = calc_current(params['simulation'], an, ca)
    
    # Figure out which steps and at what currents to run the model. This 
    # returns a tuple of 'charge' and 'discharge' steps, and a tuple with a 
    # current for each step.
    steps, currents = setup_cycles(params['simulation'], current)

    def terminate_check(t, SV, SVdot, return_val, inputs):
        return_val[0] = ca.voltage_lim(SV, ca, params['simulation']
                ['phi-cutoff-lower'])
        return_val[1] = ca.voltage_lim(SV, ca, params['simulation']
                ['phi-cutoff-upper'])

    # Set up the solver:
    options =  {'user_data':(an, sep, ca, params), 'rtol':1e-8, 'atol':1e-11, 
            'algebraic_vars_idx':algvars, 'first_step_size':1e-18, 
            'rootfn':terminate_check, 'nr_rootfns':2}
    solver = dae('ida', residual, **options)

    for i, step in enumerate(steps):
        print(step,'...\n')

        # Set the external current density (A/m2)
        params['i_ext'] = currents[i]
        print('    Current = ', round(currents[i],3),'\n')
        t_out = np.linspace(0,t_final,10000)

        # Make the initial solution consistent with the algebraic constraints:
        SV_0 = an.make_alg_consistent(SV_0, an, sep, ca, params)
        SV_0 = sep.make_alg_consistent(SV_0, an, sep, ca, params)
        
        # This runs the integrator.
        SVdot_0 = np.zeros_like(SV_0)
        solution = solver.solve(t_out, SV_0, SVdot_0)

        # Create an array of currents, one for each time step:
        i_data = currents[i]*np.ones_like(solution.values.t)

        # Append the current data array to any preexisting data, for output.  
        # If this is the first step, create the output data array.
        if i:
            # Stack the times, the current at each time step, and the solution 
            # vector at each time step into a single data array.
            SV = np.vstack((solution.values.t+data_out[0,-1], i_data, solution.
                values.y.T))
            data_out = np.hstack((data_out, SV))

            # Use SV at the end of the simualtion as the new initial condition:
            SV_0 = solution.values.y[-1,:]
        else:
            # Stack the times, the current at each time step, and the solution 
            # vector at each time step into a single data array.
            SV = np.vstack((solution.values.t, i_data, solution.values.y.T))
            data_out = SV

            # Use SV at the end of the simualtion as the new initial condition:
            SV_0 = solution.values.y[-1,:]

    return data_out

def calc_current(params, an, ca):
    # Calculates the external current from the user inputs.  If a C-rate is 
    # given, calculate the battery capacity and convert this to a current.  If 
    # i_ext is given, convert the units to A/m2.

    # Battery capacity is the lesser of the anode and cathode capacities. It is 
    # required for determining the simulation time.
    cap = min(an.capacity, ca.capacity)
    if params['i_ext'] is not None:
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

            # Read the units and convert i_ext as necessary:
            i_units, A_units = units.split('/')
            if i_units=="mA":
                i_ext *= 0.001
            elif i_units=="uA":
                i_ext *= 1e-6
            if A_units=="cm2":
                i_ext *= 10000
    elif params['C-rate'] is not None:

        i_ext = cap*params['C-rate']

    else:
        # If neither i_ext or C_rate is provided, throw an error:
        raise ValueError("Please specify either the external current (i_ext) "
            "or the C-rate (C-rate).")
    
    t_final = cap*3600/i_ext

    return i_ext, t_final

def setup_cycles(params, current):
    # Setup a tuple representing steps in the requested charge-discharge cycles.
    # Also create a tuple of currents, one for each step.
    steps = ()
    currents = ()

    if params['first-step'] == "discharge":
        cycle = ('discharge', 'charge')
        cycle_currents = (current, -current)
    else:
        cycle = ('charge','discharge')
        cycle_currents = (-current, current)
    
    # At present, the only partial cycle accepted is for a single half-cycle 
    # (i.e. a single charge or discharge step).
    #TODO #16
    if params['n_cycles'] < 1.0:
        steps = (cycle[0],)
        currents = (cycle_currents[0],)
    else:
        steps = params['n_cycles']*cycle
        currents = params['n_cycles']*cycle_currents

    return steps, currents

def residual(t, SV, SVdot, resid, inputs):
    # Call the individual component residual functions, which implement 
    # governing equations to solve for the state at any given time step. 
    # Nothing is returned by this function. It merely needs to set the value of 
    # 'SVdot':
    an, sep, ca, params = inputs

    # Call residual functions for anode, separator, and cathode:
    resid[an.SVptr['residual']] = an.residual(SV, SVdot, an, sep, ca, params)

    resid[sep.SVptr['residual']] = sep.residual(SV, SVdot, an, sep, ca, params)
    
    resid[ca.SVptr['residual']] = ca.residual(SV, SVdot, ca, sep, an, params)

def output(solution, an, sep, ca, params):
    # Prepare and save any output data to the correct location. Prepare, 
    # create, and save any figures relevant to constant-current cycling.
    #TODO #17
    import matplotlib.pyplot as plt 

    # Calculate cell potential:   
    V_cell = solution[2+ca.SV_offset+ca.SVptr['phi_ed'],:]

    # Create figure:
    lp = 30 #labelpad
    fig, axs = plt.subplots(3,1, sharex=True, 
            gridspec_kw = {'wspace':0, 'hspace':0})
    axs[0].plot(solution[0,:]/3600, 1000*solution[1,:]/10000)
    axs[0].set_ylabel('Current Density \n (mA/cm$^2$)',labelpad=lp-25)
    axs[1].plot(solution[0,:]/3600, V_cell)
    axs[1].set_ylabel('Cell Potential \n(V)',labelpad=lp)
    axs[2].plot(solution[0,:]/3600, 1e6*solution[2+an.SVptr['thickness']])
    axs[2].set_ylabel('Anode Thickness \n($\mu$m)', labelpad=lp-10)
    axs[2].set(xlabel='Time (h)')
    # plt.figure(2)
    # plt.plot(solution[0,:]/3600, solution[2+an.SVptr['phi_dl']])
    for i in range(3):
        axs[i].tick_params(axis="x",direction="in")
        axs[i].tick_params(axis="y",direction="in")
    fig.tight_layout()
    plt.show()
