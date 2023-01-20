
import matplotlib.pyplot as plt
import numpy as np

class voltage_capacity:

    def SSR(ref, sim, units_scale = 1.0):
        residual = 0

        # Read out simulation data:
        sim_caps = sim[:,0]
        sim_voltages = sim[:,1]

        # Read out reference data
        ref_caps = ref[:,0]*units_scale
        ref_voltages = ref[:,1]

        # Calculate the average dx in capacity of the reference data
        ref_cap_dx_avg = np.average(np.ediff1d(ref_caps))

        # If the max capacity of the reference data and the model data is not
        #   the same, append an array of capacity values using the average
        #   change in capacity of the reference data to the shorter capacity
        #   data. Append an array of zeros_like the capacity data appended to
        #   the voltage data of the matching capacity data
        if ref_caps[-1] > sim_caps[-1]:
            # Calculate the cap values to append to the simulation data
            caps_append = np.arange(sim_caps[-1]*1.001, ref_caps[-1],
                                    ref_cap_dx_avg)
            caps_append = np.append(caps_append, ref_caps[-1])
            sim_caps = np.append(sim_caps, caps_append)

            # Create array of zeros to append to the sim voltage
            voltages_append = np.zeros_like(caps_append)
            sim_voltages = np.append(sim_voltages, voltages_append)
        elif sim_caps[-1] > ref_caps[-1]:
            # Calculate the cap values to append to the reference data
            caps_append = np.arange(ref_caps[-1]*1.001, sim_caps[-1],
                                    ref_cap_dx_avg)
            caps_append = np.append(caps_append, sim_caps[-1])
            ref_caps = np.append(ref_caps, caps_append)

            # Create array of zeros to append to the ref voltage
            voltages_append = np.zeros_like(caps_append)
            ref_voltages = np.append(ref_voltages, voltages_append)

        # Interpolate to find the simulated voltage for each reported capacity
        #   from the reference data
        sim_voltages_int = np.interp(ref_caps, sim_caps, sim_voltages)

        for i, ref_cap in enumerate(ref_caps):
            # if it's the first capacity value or the difference in capacity
            #   between the current and last value is greater than the average
            #   change in capacity of the reference data, calculate the SSR at
            #   the current point and add it to the total SSR residual value
            if i == 0:
                cap_prev = ref_cap
                residual += (ref_voltages[i] - sim_voltages_int[i])**2
            else:
                d_cap = ref_cap - cap_prev
                cap_prev = ref_cap
                if d_cap >= ref_cap_dx_avg:
                    residual += (ref_voltages[i] - sim_voltages_int[i])**2

        return residual

    def plot(ref, sim, ax, fig, units_scale = 1.0, color = 'k'):
        sim_voltage = np.zeros_like(ref[:,1])
        ref_cap = np.zeros_like(ref[:,1])

        for i, row in enumerate(ref):
            ref_cap[i] = row[0]*units_scale

            # Interpolate to find the simulated voltage for each reported
            # capacity from the reference data:
            sim_voltage[i] = np.interp(ref_cap[i], sim[:,0], sim[:,1])

        ax.plot(sim[:,0], sim[:,1], color=color)
        ax.plot(ref_cap, ref[:,1], color = color, marker='x', linestyle=None)

        return ax, fig
        # summary_axs[0].set_ylabel('Current Density \n (mA/cm$^2$)',labelpad=lp
