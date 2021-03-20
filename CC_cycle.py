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
from scikits.odes.ode import ode

def run(SV_0, an, sep, ca, params):
    # Determine the current to run at. 'calc_current' is defined below.
    current, t_final = calc_current(params['simulation'], an, ca)
    
    # Figure out which steps and at what currents to run the model. This 
    # returns a tuple of 'charge' and 'discharge' steps, and a tuple with a 
    # current for each step.
    steps, currents = setup_cycles(params['simulation'], current)

    # Set up the solver:
    options =  {'user_data':(an, sep, ca, params), 'rtol':1e-6, 'atol':1e-8}
    solver = ode('cvode', residual, **options)

    for i, step in enumerate(steps):
        print(step,'...\n')

        # Set the external current density (A/m2)
        params['i_ext'] = currents[i]
        print('     Current = ', round(currents[i],3),'\n')
        
        t_out = np.linspace(0,t_final) # TEMPORARY
    
        # This runs the integrator. The 'residual' function is defined below.
        solution = solver.solve(t_out, SV_0)

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
    print(an.capacity)
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
        # To be implemented.
        #TODO #15
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

def residual(t,SV, SVdot, inputs):
    # Call the individual component residual functions, which implement 
    # governing equations to solve for the state at any given time step. 
    # Nothing is returned by this function. It merely needs to set the value of 
    # 'SVdot':
    an, sep, ca, params = inputs

    # Call residual functions for anode, separator, and cathode:
    SVdot[an.SVptr['residual']] = an.residual(SV, an, params)

    SVdot[sep.SVptr['residual']] = sep.residual(SV, sep, params)
    
    SVdot[ca.SVptr['residual']] = ca.residual(SV, ca, params)

def output(solution, an, sep, ca, params):
    # Prepare and save any output data to the correct location. Prepare, 
    # create, and save any figures relevant to constant-current cycling.
    #TODO #17
    import matplotlib.pyplot as plt 

    # Calculate cell potential:   
    V_cell = (solution[2+an.SVptr['phi_dl'],:] 
            - solution[2+ca.SV_offset+ca.SVptr['phi_dl'],:])

    # Create figure:
    fig, axs = plt.subplots(2,1, sharex=True)
    axs[0].plot(solution[0,:]/3600, 1000*solution[1,:]/10000)
    axs[0].set(ylabel='Current Density (mA/cm$^2$)')
    axs[1].plot(solution[0,:]/3600, V_cell)
    axs[1].set(ylabel='Cell Potential (V)', xlabel='Time (h)')
    fig.tight_layout
    plt.show()
