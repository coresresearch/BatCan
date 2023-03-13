# bat_can_fit.py
"""
    BatCan_Fit - Battery Modeling in Cantera

    This file is for fitting modeling results to user-specified reference data.

    This file reads in the user input, runs the simulation, and then produces any requested output (saved data file, preliminary plots, etc.)

"""
# Import modules
from datetime import datetime   
from functools import partial
import importlib # allows us to import from user input string.
import itertools
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import pandas as pd
from scipy import optimize as spo
import timeit

#todo #53
from submodels.fitting import voltage_capacity as fit
from bat_can_init import initialize

# This is the main function that runs the model.  We define it this way so it 
# is called by "main," below:
def bat_can(input, cores, print_flag):
    if input is None:
        # Default is a single-particle model of graphite/LCO
        input_file = 'inputs/spmGraphite_PorousSep_spmLCO_input.yaml'
    else:
        if input[-5:] == '.yaml':
            input_file  = 'inputs/'+input

            # Strip the file extension:
            input = input[:-4]
        else:
            input_file = 'inputs/'+input+'.yaml'

    if cores is None:
        cores = 1
    #===========================================================================
    #   READ IN USER INPUTS
    #===========================================================================
    an_inputs, sep_inputs, ca_inputs, parameters, fit_params = \
        initialize(input_file, fit = True)

    now = datetime.now()
    dt =  now.strftime("%Y%m%d_%H%M")

    # Save name of input file, without path or extension:
    parameters['input'] = input
    parameters['output'] = 'outputs/' + parameters['input']+ '_' + dt
    #===========================================================================
    #   CREATE ELEMENT CLASSES AND INITIAL SOLUTION VECTOR SV_0
    #===========================================================================
    # For each element (anode 'an', separator 'sep', cathode 'ca') the 'class' 
    # variable from the inputs tells what kind of anode, separator, or cathode 
    # it is, and points to a '.py' file in this directory.  We import that 
    # module, and then run its 'initialize' routine to create an intial 
    # solution vector and an object that stores needed parameters.
    # import single_particle_electrode as an_module_0
    
    an_module = importlib.import_module('electrode_models.' 
        + an_inputs['class'])
    an = an_module.electrode(input_file, an_inputs, sep_inputs, ca_inputs, 
            'anode', parameters, offset=0)
    
    sep_module = importlib.import_module('separator_models.' 
        + sep_inputs['class'])
    sep = sep_module.separator(input_file, sep_inputs, parameters, 
            offset=an.n_vars)
    
    # Check to see if the anode object needs to adjust the separator properties:
    sep = an.adjust_separator(sep)
    ca_module = importlib.import_module('electrode_models.' 
        + ca_inputs['class'])
    ca = ca_module.electrode(input_file, ca_inputs, sep_inputs, an_inputs, 
        'cathode', parameters, offset= an.n_vars+sep.n_vars*sep.n_points)

    # Check to see if the cathode object needs to adjust the separator 
    # properties:    
    sep = ca.adjust_separator(sep)

    # Initialize the solution vector:
    SV_an_0 = an.initialize(an_inputs, sep_inputs)
    SV_sep_0 = sep.initialize(sep_inputs)
    SV_ca_0 = ca.initialize(ca_inputs, sep_inputs)

    # Stack the three initial solution vectors into a single vector:
    SV_0 = np.hstack([SV_an_0, SV_sep_0, SV_ca_0])
    # Ditto for the algebraic variable indices:
    algvars = np.hstack([an.algvars, sep.algvars, ca.algvars])

    #===========================================================================
    #   RUN THE SIMULATION
    #===========================================================================
    # The inputs tell us what type of experiment we will simulate.  Load the 
    # module, then call its 'run' function:

    # If the user requests a specific initailization routine, run that first:
    if 'initialize' in parameters and parameters['initialize']['enable']:
        if parameters['initialize']['type'] == 'open-circuit':
            model = importlib.import_module('.'+'CC_cycle', 
                package='simulations')
            
            sim = {'i_ext': None, 'C-rate': 0.0, 'n_cycles': 0, 
                'first-step': 'discharge', 'equilibrate': 
                {'enable': True, 'time':  5}, 'phi-cutoff-lower': 2.0, 
                'phi-cutoff-upper': 4.8, 'init':True}
            
            solution = model.run(SV_0, an, sep, ca, algvars, parameters, sim)
            
            # Save final state as the initial state for all subsequent 
            # simulation steps:
            SV_0 = model.final_state(solution)

        else:
            raise ValueError("Initialization method currently not implemented.")

    def run_model(x_guess, final_flag = False):
        print('x = ',x_guess)
    
        # Set the current fitting parameter values:
        for i, x in enumerate(x_guess):
            if fit_params[i]['type'] == 'elyte-transport':
                sep.D_k[sep.elyte_obj.species_index(fit_params[i]['species'])] \
                    = x
            elif fit_params[i]['type'] == 'cathode-kinetics':
                ca.surf_obj.set_multiplier(x, 
                    fit_params[i]['reaction-index'])
            elif fit_params[i]['type'] == 'air-kinetics':
                ca.gas_elyte_obj.set_multiplier(x, 
                    fit_params[i]['reaction-index'])
            elif fit_params[i]['type'] == 'cathode-microstructure':
                setattr(ca, fit_params[i]['parameter'], x)

        if final_flag:
            SSR_net = 0
            icolor = 0
            fit_fig, fit_axs = plt.subplots(1, 1, sharex=True, 
                gridspec_kw = {'wspace':0, 'hspace':0})
        
            fit_fig.set_size_inches((5.0, 2.25))

            # pool = mp.Pool(processes = int(cores))
            # SSR_net = pool.map(run, (list(parameters['simulations'])))
            for sim in parameters['simulations']:
                SSR_net += run(sim, SV_0, algvars, parameters, an, ca, sep, 
                               final_flag, fit_fig, fit_axs, icolor)
                icolor += 1
        else:
            with Pool(processes = int(cores)) as pool:
                SSR_net = pool.map(partial(run, SV_0=SV_0, algvars=algvars, 
                            parameters=parameters, an=an, sep=sep, ca=ca), 
                            list(parameters['simulations']))

        print('SSR = ', np.sum(SSR_net))
        
        if final_flag:
            fit_axs.annotate(f"SSR = 1.0", xy=(0,0))#,
                # xytext=(0.5, 0.5), textcoords='axes fraction')#, fontsize = 8)
                
            if len(parameters['simulations']) == 1:
                savename = (parameters['output'] +'_' 
                    + sim['outputs']['save-name'] )
            else:
                savename = (parameters['output'])
                
            fit_fig.savefig(savename + '/fit.pdf')
            plt.show()

        return np.sum(SSR_net)

    start = timeit.default_timer()
    x_start = []
    x_bounds = []
    for x in fit_params:
        x_start.append(x['guess-value'])
        x_bounds.append((x['min'], x['max'])) #todo #54

    for sim in parameters['simulations']:
        sim['ref_data'] = pd.read_excel('data/' + sim['validation'])

    
    if print_flag:
        # Just print the fit of the starting guess - do not run the fitting 
        # routine:
        result = run_model(x_start, final_flag=True)
    else:
        # Fit to the reference data:
        result = spo.minimize(run_model, x_start, bounds = x_bounds, 
                options={'disp': True})
    
        print(result)

        print("Best fit = ", result.x)

        run_model(result.x, final_flag=True)

    stop = timeit.default_timer()
    print('Time: ', stop - start)  

def run(sim, SV_0, algvars, parameters, an, ca, sep, final_flag=False, 
            fit_fig=None, fit_axs=None, icolor=0):
    
    sim_local = sim
    try:
        model = importlib.import_module('.'+sim['type'], 
            package='simulations')

        sim['init'] = False
        solution = model.run(SV_0, an, sep, ca, algvars, 
            parameters, sim)
        # Read out the results:
        if final_flag:
            sim_local['outputs']['show-plots'] = False
            results = model.output(solution, an, sep, ca, parameters, 
                sim_local, plot_flag=True, return_flag=True, save_flag=True)
                #     for sim in parameters['simulations']:
                # model = importlib.import_module('.'+sim['type'], package='simulations')
            solution = model.plot(an, sep, ca, parameters, sim_local)
        else:
            results = model.output(solution, an, sep, ca, parameters, 
                sim, plot_flag=False, return_flag=True, save_flag=False)
        phi_sim = results['phi_ed'].to_numpy()[:,-1]
            
        sim_data = np.array((results['capacity'].to_numpy(),
            phi_sim))

        # This function calculates the SSR for this simulation:
        # Scale to convert capacity units in validation data to mAh/cm2:
        UnitsScale = 46.968 #todo #72
        ssr_calc = fit.SSR(sim['ref_data'].to_numpy(), sim_data.T, 
                units_scale = UnitsScale)
        print('SSR = ', ssr_calc)
        
        ndata = len(parameters['simulations']) + 1
        cmap = plt.get_cmap('plasma')
        
        color_ind = np.linspace(0,1,ndata)
        colors = list()
        
        for i in np.arange(ndata):
            colors.append(cmap(color_ind[i]))
  
        if final_flag:
            fit_axs, fit_fig = fit.plot(sim['ref_data'].to_numpy(), 
                sim_data.T, fit_axs, fit_fig, units_scale = UnitsScale, 
                color = colors[icolor])
            
    except:
        # Assign a large penalty for failed parameter sets:
        ssr_calc = 1e23
        print('Simulation failed')

    return ssr_calc



#===========================================================================
#   FUNCTIONALITY TO RUN FROM THE COMMAND LINE
#===========================================================================
if __name__ == '__main__':
    import argparse

    # Currently, the only command line keyword enabled is --input, to specify 
    # the input file location:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--cores')
    parser.add_argument('--print', action='store_true')
    args = parser.parse_args()
    
    bat_can(args.input, args.cores, args.print)
