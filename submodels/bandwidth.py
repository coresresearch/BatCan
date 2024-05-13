"""submodules.bandwidth - module to calculate bandwidth for a banded linsolver
"""
import cantera as ct
import numpy as np

def calc_bandwidth(SV_0, an, sep, ca, params):
    # Calculate size N of Jacobian
    N = np.size(SV_0)
    lband = 0
    uband = 0

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

    jac = np.zeros([N, N])
    resid_0 = calc_resid(SV_0)

    for i in range(N):
        dSV = np.copy(SV_0)
        dSV[i] = 1.01 * (SV_0[i] + 0.01)
        dF = resid_0 - calc_resid(dSV)
        jac[:,i] += dF
    
    # Scan rows for max lband/uland
    for i in range(N):
        l_inds = np.where(abs(jac[i, :i]) > 0.)[0]
        if len(l_inds) >= 1 and i - l_inds[0] > lband:
            lband = i - l_inds[0]

        u_inds = i + np.where(abs(jac[i, i:]) > 0.)[0]
        if len(u_inds) >= 1 and u_inds[-1] - i > uband:
            uband = u_inds[-1] - i

    return lband, uband
