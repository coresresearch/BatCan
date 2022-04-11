import numpy as np

def SSR_voltage_capacity(ref, sim):
    residual = 0
    for row in ref:
        ref_voltage = row[1]
        sim_voltage = np.interp(row[0], sim[:,0],sim[:,1])

        residual += (ref_voltage - sim_voltage)**2

    return residual

