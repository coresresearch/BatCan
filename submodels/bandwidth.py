"""submodules.bandwidth - module to calculate bandwidth for a banded linsolver
"""
import cantera as ct
import numpy as np

def calc_bandwidth(SV_0, an, sep, ca, params):
    # Calculate size N of Jacobian
    N = np.size(SV_0)
    lband = 1
    uband = 1
    SVdot = np.zeros_like(SV_0)
    params['i_ext'] = 0

    def calc_resid(SV):
        # Call residual functions for anode, separator, and cathode. Assemble them
        # into a single residual vector 'resid':
        resid_i = np.zeros_like(SV)
        resid_i[an.SVptr['electrode']] = an.residual(0, SV, SVdot, sep, ca, params)

        resid_i[sep.SVptr['sep']] = sep.residual(SV, SVdot, an, ca, params)

        resid_i[ca.SVptr['electrode']] = ca.residual(0, SV, SVdot, sep, an, params)
        return resid_i

    resid_0 = calc_resid(SV_0)
    for i in range(N):
        for j in range(N):
            dSV = np.copy(SV_0)
            dSV[j] = 1.01*SV_0[j]
            dF = resid_0 - calc_resid(dSV)
            if abs(dF[i]) > 0:
                if j > i and abs(i - j) > uband:
                    uband = abs(i - j)
                elif i > j and abs(i - j) > lband:
                    lband = abs(i - j)
    return lband, uband
