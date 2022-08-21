# bat_can_init.py
"""
    BatCan - Battery Modeling in Cantera

    Initialize the battery simulation.

    Initialize variables, parameters, and Cantera objects for the simulation.

    - Reads user inputs.
    - Creates necessary Cantera objects.
    - Reads and stores necessary parameters.
    - Initializes the solution vector


    Returns: 
    - Initial solution vector SV_0
    - Dictionary of cantera objects obj
    - Dictionary of paramters params
    - Dictionary of pointers which allow the simulation to located needed terms.
    """

def initialize(input_file=None, fit=False):
    
    # Import necessary modules:
    from ruamel.yaml import YAML
    from pathlib import Path
    import cantera as ct
    import numpy as np
    import importlib

    """
    =========================================================================
       READ IN INPUTS
    =========================================================================
    """
    path = Path(input_file)
    yaml = YAML(typ='safe')
    inputs = yaml.load(path)

    #===========================================================================
    #   LOAD ANODE / WORKING ELECTRODE
    #===========================================================================
    if 'anode' in inputs['cell-description']:
        anode_inputs = inputs['cell-description']['anode']
    else:
        raise ValueError('Please specify an anode type.')
    
    #===========================================================================
    #   LOAD SEPARATOR
    #===========================================================================
    if 'separator' in inputs['cell-description']:
        sep_inputs = inputs['cell-description']['separator']
    else:
        # Generic separator:
        sep_inputs = 'separator'
    
    #===========================================================================
    #   LOAD CATHODE / COUNTER ELECTRODE
    #===========================================================================
    if 'cathode' in inputs['cell-description']:
        cathode_inputs = inputs['cell-description']['cathode']
    else:
        raise ValueError('Please specify a cathode type.')

    #===========================================================================
    #   LOAD SIMULATION PARAMETERS
    #===========================================================================
    if 'parameters' in inputs:
        parameters = inputs['parameters']
        # Read T and P and convert units to K and Pa, as necessary:
        parameters['T'], parameters['P'] = read_conditions(parameters)
    else:
        raise ValueError('Please specify simulation parameters.')

    if fit:
        fit_parameters = inputs['fit-parameters']
        return (anode_inputs, sep_inputs, cathode_inputs, parameters, 
             fit_parameters)
    else:
         return anode_inputs, sep_inputs, cathode_inputs, parameters

def read_conditions(params):
    # Read the pressure from the input paramters and convert units to Pa, 
    # if necessary.
    
    # Read in the user inputs:
    temperature, pressure = params['T'], params['P']

    # Split the values from the units:
    pres, p_units = pressure.split()
    temp, T_units = temperature.split()

    # Convert values to floats:
    temp = float(temp)
    pres = float(pres)

    # Read the temperature units and convert to K as necessary:
    if T_units=="C":
        temp += 273.15
    elif T_units=="F":
        temp = (temp + 459.67)* 5. / 9.
    elif T_units!="K":
        ValueError("Please provide temperature in an accepted unit:", 
            " C, K, or F.")
    

    # Read the pressure units and convert to Pa as necessary:
    if p_units=="atm":
        pres *= 101325
    elif p_units=="kPa":
        pres *= 1.e3
    elif p_units=="MPa":
        pres *= 1.e6
    elif p_units=="bar":
        pres *= 1.e5
    elif p_units!="Pa":
        ValueError("Please provide pressure in an accepted unit: Pa, kPa,", 
            " MPa, atm, or bar.")

    return temp, pres
   