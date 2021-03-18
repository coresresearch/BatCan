"""
    CC_cycle.py

    Functions and outputs for constant-current cycling
"""
import numpy as np
from scikits.odes.odeint import odeint

def run(SV_0, an, sep, ca, params):
    # Determine the current to run at:
    current = calc_current(params['simulation'], an, ca)
    
    # Figure out which steps and at what currents to run the model. This 
    # returns a tuple of 'charge' and 'discharge' steps, and a tuple with a 
    # current for each step.
    steps, currents = setup_cycles(params['simulation'], current)
    i_step = 0

    for i, step in enumerate(steps):
        print(step,'...\n')
        params['i_ext'] = currents[i]
        print('     Current = ', round(currents[i],3),'\n')
        
        t_out = np.linspace(0,1) # TEMPORARY
    
        solution = odeint(residual, t_out, SV_0)
    
        if i_step:
            SV_data = np.vstack((SV_data, solution.values.y))
            t_data = np.hstack((t_data, t_data[-1]+solution.values.t))
        else:
            SV_data = solution.values.y
            t_data = solution.values.t

        i_step += 1    

    return t_data, SV_data

def calc_current(params, an, ca):
    if params['i_ext'] is not None:
        if params['C-rate'] is not None:
            raise ValueError("Both i_ext and C-rate are specified. "
                "Please specify only one of the two in your input file.")
        else:
            current = parse_current(params['i_ext'])
        
    return current

def parse_current(current):
    # Isolate the current
    i_ext, units = current.split()
    # Convert i_ext to a float:
    i_ext = float(i_ext)

    # Read the units and convert i_ext as necessary:
    i, area = units.split('/')
    if i=="mA":
        i_ext *= 0.001
    if area=="cm2":
        i_ext *= 10000

    return i_ext

def setup_cycles(params, current):
    steps = ()
    currents = ()

    if params['first-step'] == "discharge":
        cycle = ('discharge', 'charge')
        cycle_currents = (current, -current)
    else:
        cycle = ('charge', 'discharge')
        cycle_currents = (-current, current)
    
    if params['n_cycles'] < 1.0:
        steps = (cycle[0],)
        currents = (cycle_currents[0],)
    else:
        steps = params['n_cycles']*cycle
        currents = params['n_cycles']*cycle_currents

    return steps, currents

def residual(t,SV, SVdot):
        resid = np.zeros_like(SV)
        SVdot = resid

def output(solution):
    pass
