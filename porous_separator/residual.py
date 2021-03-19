"""
    porous_separator.py

    Class file for porous separator methods
"""

import cantera as ct
import numpy as np

def residual(SV, self, params):
    SV_dot = np.zeros((self.nVars,))

    return SV_dot