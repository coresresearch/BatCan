# mech_test.py
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

h_values = [0, 10, 20, 30]
fig, ax = plt.subplots(2,1, gridspec_kw = {'wspace':0, 'hspace':0})

cmap = plt.get_cmap('plasma')
ndata = len(h_values)+1
color_ind = np.linspace(0,1,ndata)
colors = list()
for i in np.arange(ndata):
    colors.append(cmap(color_ind[i]))

legends = []
for i, h in enumerate(h_values):
    file =  'Li_metal_spm_LCO_input_h'+str(h)+'.yaml'

    li = ct.Solution(file,'lithium_metal')
    elyte = ct.Solution(file,'electrolyte')
    conductor = ct.Solution(file,'electron')
    surf = ct.Interface(file,'lithium_electrolyte',[li, elyte, conductor])

    phi_array  = np.linspace(-3.5,-2.5,500)
    currents = np.zeros_like(phi_array)
    q_fwd = np.zeros_like(phi_array)
    q_rev = np.zeros_like(phi_array)

    legends.append('h$^\circ_\mathrm{Li}$ = '+str(h)+' kJ/mol' )
    for j, phi in enumerate(phi_array):
        li.electric_potential = phi
        conductor.electric_potential = phi
        elyte.electric_potential = 0
        currents[j] = ct.faraday*surf.get_net_production_rates(conductor)[0]/10000
        q_fwd[j] = surf.get_creation_rates(conductor)[0]
        q_rev[j] = surf.get_destruction_rates(conductor)[0]


    ax[0].semilogy(phi_array, abs(currents),color=colors[i])
    ax[0].set(ylabel='Abs. Value \n Current Density (A cm$^{-2}$)')
    ax[1].semilogy(phi_array, q_fwd, color=colors[i])
    ax[1].semilogy(phi_array, q_rev, color=colors[i])
    ax[1].set(xlabel='Electric potential difference (V)',
            ylabel='Rate of progress\n (kmol s$^{-1}$m$^{-2}$)')

ax[0].get_xaxis().set_ticklabels([])
ax[0].legend(legends,loc='lower right',frameon=False)

ax[0].tick_params(axis="y",direction="in")
ax[0].tick_params(axis="x",direction="in")
ax[1].tick_params(axis="y",direction="in")
ax[1].tick_params(axis="x",direction="in")
plt.subplots_adjust(wspace=0, hspace=0)

fig.tight_layout()
plt.savefig('anode_thermo_test.png',dpi=300)
plt.show()