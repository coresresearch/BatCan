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
import time
from scikits.odes.dae import dae
from math import floor
from submodels import bandwidth

def run(SV_0, an, sep, ca, algvars, params, sim):
    """
    Run the simulation
    """
    #t_count = time.time()
    # Determine the current to run at, and the time to fully charge/discharge.
    # 'calc_current' is defined below.
    current, t_final = calc_current(sim, an, ca)

    # Store the location of all algebraic variables.
    params['algvars'] = algvars

    # Concatenate atol arrays from each component
    atol_vec = np.hstack([an.atol, sep.atol, ca.atol])

    # Concatenate constr_type into an array
    constr_type = np.hstack([an.constraints_type, sep.constraints_type,
                                  ca.constraints_type])

    # Specify the boundary condition as galvanostatic:
    params['boundary'] = 'current'

    # Figure out which steps and at what currents to run the model. This
    # returns a tuple of 'charge' and 'discharge' steps, and a tuple with a
    # current for each step. 'equil' is a flag to indicate whether there is an
    # equilibration step.
    steps, currents, times, equil = setup_cycles(sim, current, t_final)
    n_steps = len(steps)

    # Calculate the bandwidth used for the band linsolver
    lband, uband = bandwidth.calc_bandwidth(SV_0, an, sep, ca, params)
    print('lband =', lband, 'uband =', uband)
    #print(stop_sim)

    # This function checks to see if certain limits are exceeded which will
    # terminate the simulation:
    n_roots = 2
    if 'species-cutoff' in sim:
        n_roots += 3

    def terminate_check(t, SV, SVdot, return_val, inputs):
        return_val[0] = ca.voltage_lim(SV, sim['phi-cutoff-lower'])
        return_val[1] = ca.voltage_lim(SV, sim['phi-cutoff-upper'])

        if 'species-cutoff' in sim:
            return_val[2] = an.species_lim(SV, sim['species-cutoff'])
            return_val[3] = sep.species_lim(SV, sim['species-cutoff'])
            return_val[4] = ca.species_lim(SV, sim['species-cutoff'])

    # Set up the differential algebraic equation (dae) solver:
    options =  {'user_data':(an, sep, ca, params), 'rtol':1e-3, 'atol':atol_vec,
            'algebraic_vars_idx':algvars, 'first_step_size':1e-18, 'max_step_size':1,
            'rootfn':terminate_check, 'nr_rootfns':n_roots, 'compute_initcond':'yp0',
            'constraints_type':constr_type, 'linsolver':'band', 'lband':lband, 'uband':uband}

    solver = dae('ida', residual, **options)

    # Go through the current steps and integrate for each current:
    for i, step in enumerate(steps):
        print('Step ',int(i+1),'(out of', n_steps, '): ',step,'...\n')

        # Set the external current density (A/m2)
        params['i_ext'] = currents[i]
        print('    Current = ', round(currents[i],3),'A/m^2 \n')

        t_out = np.linspace(0, times[i], 10000)

        # Create an initial array of time derivatives and runs the integrator:
        SVdot_0 = np.zeros_like(SV_0)
        solution = solver.solve(t_out, SV_0, SVdot_0)

        # Create an array of currents, one for each time step:
        i_data = currents[i]*np.ones_like(solution.values.t)
        cycle_number = int(i+1-equil)*np.ones_like(solution.values.t)
        cycle_capacity = solution.values.t*abs(i_data)/3600/ca.m_S_tot_0 #1000*solution.values.t*abs(i_data)/3600

        # Append the current data array to any preexisting data, for output.
        # If this is the first step, create the output data array.
        if i: # Not the first step. 'data_out' already exists:
            # Stack the times, the current at each time step, and the solution
            # vector at each time step into a single data array.
            SV = np.vstack((solution.values.t+data_out[0,-1], cycle_number,
                i_data, cycle_capacity, solution.values.y.T))
            data_out = np.hstack((data_out, SV))

            # Use SV at the end of the simualtion as the new initial condition:
            SV_0 = solution.values.y[-1,:]

        else: # First step. 'data_out' does not yet exist:
            # Stack the times, the current at each time step, and the solution
            # vector at each time step into a single data array.
            SV = np.vstack((solution.values.t, cycle_number, i_data,
                cycle_capacity, solution.values.y.T))

            data_out = SV

            # Use SV at the end of the simualtion as the new initial condition:
            SV_0 = solution.values.y[-1,:]

    #t_elapsed = time.time() - t_count
    #print('t_cpu =', t_elapsed, '\n')

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
    if i_ext > 0:
        t_final = cap*3600/i_ext
    else:
        # For an equilibration/ocv run, just run for a long time:
        t_final = 1e6

    return i_ext, t_final

def setup_cycles(params, current, time):
    """
    Set up a tuple representing steps in the requested charge-discharge cycles.
    Also create a tuple of currents, one for each step.
    """
    steps = ()
    currents = ()
    times = ()
    equil = 0

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
        equil = 1
        steps = ('equilibrate',)+ steps
        currents = (0,) + currents
        times = (params['equilibrate']['time'],) + times

    return steps, currents, times, equil

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

def conservation_test(solution, an, sep, ca, params, sim):

    print('Running test suite')
    SV_offset = 4
    ca_Ck_ptr = SV_offset + ca.SV_offset + ca.SVptr['C_k_elyte']
    sep_Ck_ptr = SV_offset + sep.SV_offset + sep.SVptr['C_k_elyte']
    an_Ck_ptr = SV_offset + an.SV_offset + an.SVptr['C_k_elyte']

    ca_eps_ptr = SV_offset + ca.SV_offset + ca.SVptr['eps_conversion']
    ca_eps_elyte = np.zeros((ca.n_points, len(solution[0])))
    for j in np.arange(0, ca.n_points):
        eps_conversion = solution[ca_eps_ptr[j,:]]
        ca_eps_elyte[j,:] = 1 - ca.eps_host - np.sum(eps_conversion, axis=0)

    n_S_atoms = np.zeros((ca.elyte_obj.n_species, 1))
    for i, species in enumerate(ca.elyte_obj.species_names):
        n_S_atoms[i] = ca.elyte_obj.n_atoms(species, 'S')

    # Calculate kmol_S/m2 planar cell area
    n_S_sep = np.zeros((sep.n_points, len(solution[0])))
    n_S_ca = np.zeros((ca.n_points, len(solution[0])))

    n_S_an = an.eps_elyte*an.dy_elyte*np.sum(n_S_atoms*solution[an_Ck_ptr[0,:]], axis=0)
    for j in np.arange(0, sep.n_points):
        n_S_sep[j,:] = sep.eps_elyte*sep.dy*np.sum(n_S_atoms*solution[sep_Ck_ptr[j,:]], axis=0)
    for j in np.arange(0, ca.n_points):
        n_S_ca[j,:] = ca_eps_elyte[j,:]*ca.dy*np.sum(n_S_atoms*solution[ca_Ck_ptr[j,:]], axis=0)

    # Calculate amount of sulfur in cathode conversion phases
    n_S_ca_solid = np.zeros((ca.n_conversion_phases, len(solution[0])))
    n_S_atoms_solid = np.zeros((ca.n_conversion_phases))
    for j, species in enumerate(ca.conversion_obj):
        for i in np.arange(0, len(species.species_names)):
            n_S_atoms_solid[j] += species.n_atoms(species.species_names[i], 'S')

    for j in np.arange(0, ca.n_conversion_phases):
        eps_conversion = solution[ca_eps_ptr[:,j]]
        n_S_ca_solid[j,:] = np.sum(n_S_atoms_solid[j]*ca.conversion_obj[j].density_mole*eps_conversion*ca.dy, axis=0)

    n_S_tot = np.sum(n_S_an,axis=0) + np.sum(n_S_sep,axis=0) + np.sum(n_S_ca,axis=0) + np.sum(n_S_ca_solid,axis=0)
    pct_err_S = 100*(n_S_tot - n_S_tot[0])/n_S_tot[0]
    print('Max percent error in sulfur conservation =', max(pct_err_S))

def output(solution, an, sep, ca, params, sim, plot_flag=True,
            return_flag=False, save_flag=True):
    """
    Prepare and save any output data to the correct location. Prepare,
    create, and save any figures relevant to constant-current cycling.
    """
    #TODO #17
    #from datetime import datetime
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    # Create figure:
    lp = 30 #labelpad
    # Number of subplots
    # (this simulation produces 2: current and voltage, vs. time):
    n_plots = 2 + an.n_plots + ca.n_plots + sep.n_plots

    # There are 4 variables stored before the state variables: (1) time (s),
    # (2) cycle number, (3) current density(A/cm2) , and (4) Capacity (mAh/cm2)
    SV_offset = 4


    # Calculate cell potential:
    phi_ptr = SV_offset + ca.SV_offset+int(ca.SVptr['phi_ed'][-1])

    if plot_flag:
        # Initialize the figure:
        summary_fig, summary_axs = plt.subplots(n_plots, 1, sharex=True,
                gridspec_kw = {'wspace':0, 'hspace':0})

        summary_fig.set_size_inches((4.0,1.8*n_plots))
        #print(solution[0,::len(solution[0])-1])
        # Axis 1: Current vs. time (h):
        x_vec = np.zeros_like(solution[0,:])
        #x_vec = params['i_ext']*np.copy(solution[0,:])/3600/ca.m_S_tot_0
        x_vec = solution[3,:]
        #print(x_vec[::len(x_vec)])
        summary_axs[0].plot(x_vec, 1000*solution[2,:]/10000)
        summary_axs[0].set_ylabel('Current Density \n (mA/cm^2)',labelpad=lp)
        #summary_axs[0].set_xlim((0, 1700))
        #summary_axs[0].set_xticks([200, 400, 600, 800, 1000, 1200, 1400, 1600])

        # Axis 2: Charge/discharge potential vs. time (h).
        summary_axs[1].plot(x_vec, solution[phi_ptr,:])
        summary_axs[1].set_ylabel('Cell Potential \n(V)')#,labelpad=lp)
        summary_axs[1].set_ylim((1.8, 2.5))

        # Add any relevant anode, cathode, and separator plots:
        summary_axs = an.output(summary_axs, solution, SV_offset, x_vec, ax_offset=2)
        summary_axs = ca.output(summary_axs, solution, SV_offset, x_vec,
            ax_offset=2+an.n_plots)
        summary_axs = sep.output(summary_axs, solution, an, ca, SV_offset, x_vec,
            ax_offset=2+an.n_plots+ca.n_plots)

        summary_axs[n_plots-1].set(xlabel='Time (h)')

        # Format axis ticks:
        for i in range(n_plots):
            summary_axs[i].tick_params(axis="x",direction="in")
            summary_axs[i].tick_params(axis="y",direction="in")
            summary_axs[i].get_yaxis().get_major_formatter().set_useOffset(False)
            summary_axs[i].yaxis.set_label_coords(-0.2, 0.5)

        # Trim down whitespace:
        summary_fig.tight_layout()

        # Initialize cycle data figure:
        cycle_fig, cycle_axs = plt.subplots(1, 1, sharex=True,
                gridspec_kw = {'wspace':0, 'hspace':0})

        cycle_fig.set_size_inches((4.0,2.0))

        # Save the solution as a Pandas dataframe:
        labels = (['cycle', 'current', 'capacity'] + an.SVnames + sep.SVnames
            + ca.SVnames)
        solution_df = pd.DataFrame(data = solution.T[:,1:],
                                    index = solution.T[:,0],
                                    columns = labels)

        solution_df.index.name = 'time (s)'

        t_0 = 0
        for i in range(int(solution[1,-1])):
            cycle = solution_df[solution_df.iloc[:,0] == i+1]
            cycle_axs.plot(1000*(cycle.index-t_0)*abs(cycle.iloc[:,1])/3600,
                cycle.iloc[:,phi_ptr-1])

            # Update time offset:
            t_0 = cycle.index[-1]

        cycle_axs.set(xlabel='Capacity (mAh/cm^2)')
        cycle_axs.set(ylabel='Cell Potential (V)')

        cycle_axs.tick_params(axis="x",direction="in")
        cycle_axs.tick_params(axis="y",direction="in")
        cycle_axs.get_yaxis().get_major_formatter().set_useOffset(False)
        cycle_axs.yaxis.set_label_coords(-0.2, 0.5)
        cycle_fig.tight_layout()

    # If no specification is given on whether to show plots, assume 'True'
    if save_flag:
        if 'outputs' not in sim:
            summary_fig.savefig('output.pdf')
            cycle_fig.savefig('cycles.pdf')
            plt.show()
        else:
            #now = datetime.now()
            #dt =  now.strftime("%Y%m%d_%H%M")
            if 'save-name' in sim['outputs']:
                if len(params['simulations']) == 1 and not params['cell-test']['enable']:
                    sim['filename'] = (params['output'] +'_'
                                        + sim['outputs']['save-name'])
                elif params['cell-test']['enable']:
                    if params['cell-test']['type'] == 'grid-analysis':
                        sim['filename'] = (params['output'] +'/'
                                            + sim['outputs']['save-name'] +'_'+
                                            str(ca.n_points)+'nodes')
                    elif params['cell-test']['type'] == 'fitting':
                        dataset_num = 1
                        for root, dirs, files in os.walk(params['output']):
                            if root == params['output']:
                                dataset_num = len(dirs) + 1
                        sim['filename'] = (params['output'] + '/'
                                        + 'dataset' + str(dataset_num))
                else:
                    sim['filename'] = (params['output'] +'/'
                                        + sim['outputs']['save-name'])

                if not os.path.exists(sim['filename']):
                    os.makedirs( sim['filename'])


                solution_df.to_pickle(sim['filename']+'/output_'
                                        + sim['outputs']['save-name'] + '.pkl')
                solution_df.to_csv(sim['filename']+'/output_'
                    + sim['outputs']['save-name'] + '.csv', sep=',')
                summary_fig.savefig(sim['filename']+'/summary_'
                    + sim['outputs']['save-name'] + '.pdf')
                cycle_fig.savefig(sim['filename']+'/cycles_'
                    + sim['outputs']['save-name'] + '.pdf')

            #if 'show-plots' not in sim['outputs'] or sim['outputs']['show-plots']:
                #plt.show()

def final_state(solution):
    # Return the state vector at the final simulation time:
    return solution[4:, -1]

def initial_state(solution):

    return solution[4:, 0]
