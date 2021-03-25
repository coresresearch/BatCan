"""
    porous_separator.py

    Class file for porous separator methods
"""

import cantera as ct
import numpy as np

def residual(SV, SVdot, an, self, ca, params):
    resid = SVdot[self.SVptr['residual']]#np.zeros((self.nVars,))

    return resid
