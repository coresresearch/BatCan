"""
    CC_cycle.py

    Functions and outputs for constant-current cycling
"""
import numpy as np
from scikits.odes.odeint import odeint

def run(SV_0, an, sep, ca, params):

    t_out = np.linspace(0,1)
    def residual(t,SV, SVdot):
        resid = np.zeros_like(SV)
        SVdot = resid

    output = odeint(residual, t_out, SV_0)

    return output

def output(solution):
    pass
