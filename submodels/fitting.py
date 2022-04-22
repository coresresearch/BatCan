
import matplotlib.pyplot as plt
import numpy as np

class voltage_capacity:

    def SSR(ref, sim, units_scale = 1.0):
        residual = 0

        # Read out simulation data:
        sim_caps = sim[:,0]
        sim_voltages = sim[:,1]

        # If reference data capacity > max simulated capacity, add zeros to the # end of simulated data:
        if ref[-1,0]*units_scale > sim_caps[-1]:
            sim_caps = np.append(sim_caps, sim_caps[-1]*1.001)
            sim_caps = np.append(sim_caps, ref[-1,0]*units_scale)

            sim_voltages = np.append(sim_voltages, 0)
            sim_voltages = np.append(sim_voltages, 0)

        for ref_cap, ref_voltage in ref:
            
            # Interpolate to find the simulated voltage for each reported 
            # capacity from the reference data:
            sim_voltage = np.interp(ref_cap*units_scale, sim_caps, sim_voltages)

            # Add the square of the difference between ref and simulated 
            # voltages to the total SSR:
            residual += (ref_voltage - sim_voltage)**2

        return residual
    
    def plot(ref, sim, ax, fig, units_scale = 1.0, color = 'k'):
        sim_voltage = np.zeros_like(ref[:,1])
        ref_cap = np.zeros_like(ref[:,1])
        
        for i, row in enumerate(ref):
            ref_cap[i] = row[0]*units_scale

            # Interpolate to find the simulated voltage for each reported 
            # capacity from the reference data:
            sim_voltage[i] = np.interp(ref_cap[i], sim[:,0], sim[:,1])

        print('loop done.')
        ax.plot(sim[:,0], sim[:,1], color=color)
        ax.plot(ref_cap, ref[:,1], color = color, marker='x')
        print('plot done.')
        return ax, fig
        # summary_axs[0].set_ylabel('Current Density \n (mA/cm$^2$)',labelpad=lp


        



