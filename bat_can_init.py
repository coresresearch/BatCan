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

def initialize(input_file=None):
    
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
    else:
        raise ValueError('Please specify simulation parameters.')

    return anode_inputs, sep_inputs, cathode_inputs, parameters
    
    
